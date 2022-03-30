import math
import plyr

import numpy as np
from scipy.special import softmax

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from nle.nethack import (
    NLE_BL_HP,
    NLE_BL_HPMAX,
    NLE_BL_TIME,
    NLE_BL_SCORE,
    NLE_BL_XP,
    NLE_BL_EXP,

    CompassDirection,
    CompassDirectionLonger,
    MiscDirection,
)

from .render import render_to_rgb


compass_to_complex = dict(zip(
    # counter-clockwise from the east, use complex values for simplicity
    ('E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE',),
    np.exp(1j * np.deg2rad((
       0,   45,  90,  135, 180,  225, 270,  315,
    ))),
))


def get_compass(proba):
    """Get the  conditional probabilities of the compass directions.
    """
    dirs = {}
    for a in (
        *CompassDirection,
        *CompassDirectionLonger,
    ):
        if a.name not in dirs:
            dirs[a.name] = 0.

        if a in proba:
            dirs[a.name] += proba[a]

    # normalize and scale the directional vectors
    C = sum(dirs.values())
    return {
        CompassDirection[k]:
        z * dirs[k] / C for k, z in compass_to_complex.items()
    }


def limits(a, b, *, rtol=0.1, atol=0.01):
    """Aesthetically expand the given limits.
    """
    lo, hi = min(a, b), max(a, b)
    return (
        lo - abs(lo) * rtol - atol,
        hi + abs(hi) * rtol + atol,
    )


def viewport(x, width, *, lo, hi):
    """Get the center of the symmetric viewport [x - width, x + width]
    that is guaranteed to stay within the specified interval [lo, hi].
    """
    c = min(max(x, lo + width), hi - width)
    return c - width, c + width


def get_ylim(x, *, rtol=0.1, atol=0.01):
    return limits(min(x), max(x), rtol=rtol, atol=atol)


def plot_series(
    main,
    *rest,
    title=None,
    ylim=None,
    xlim=None,
    yaxis=False,
    ax=None,
    **kwargs,
):
    artists = []
    ax = plt.gca() if ax is None else ax

    # plot the main series, then set the aesthetics
    artists.extend(ax.plot(main, **kwargs))
    ax.set_ylim(get_ylim(main) if ylim is None else ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if title is not None:
        ax.set_title(title)

    ax.yaxis.set_visible(yaxis)  # the precise values are irrelevant
    for x in rest:
        artists.extend(ax.plot(x))

    return artists


def draw(fig, npy, t, *, actions, artists=None, view=None):
    artists = list() if artists is None else artists
    gs = GridSpec(3, 3)

    # the adjusted duration of the episode and the actions taken in the env
    n_length = len(npy.input.fin) - int(npy.input.fin[-1])
    act_ = plyr.apply(lambda x: x[1:], npy.input.act)  # XXX a_t is act[t+1]
    rew_ = plyr.apply(lambda x: x[1:], npy.input.rew)  # XXX rew[t] is r_t

    ep = plyr.apply(lambda x: x[:n_length], npy)
    ep_t = plyr.apply(lambda x: x[t], ep)

    # compute the moving view port around `t`
    xlim = None
    if view is not None:
        if isinstance(view, float):
            view = max(1, int(n_length * view))

        if not isinstance(view, int):
            raise ValueError("`view` must be float or int.")

        xlim = limits(*viewport(t, view, lo=0, hi=n_length-1),
                      rtol=0., atol=0.25)

    # the current policy and the botl stats
    proba = dict(zip(actions, softmax(ep_t.output.pol, axis=-1).tolist()))
    blstats = ep.input.obs['blstats']

    # render the agent's view with the compass action distribution overlay
    view = ax = fig.add_subplot(gs[:2, :2])
    vicinity = render_to_rgb(ep_t.input.obs['vicinity'])
    artists.append(ax.imshow(vicinity, zorder=-99, alpha=0.75, animated=True))

    compass = get_compass(proba)
    uv = np.asarray([(z.real, z.imag) for z in compass.values()])
    cc = ['black' if actions[act_[t]] == a else 'magenta' for a in compass]

    n_row, n_col = vicinity.shape[:2]
    xy = np.full_like(uv, (n_row / 2, n_col / 2))
    artists.append(
        ax.quiver(*xy.T, *uv.T, color=cc, scale=1e-2,
                  zorder=10, angles='uv', units='xy', width=.6)
    )
    if actions[act_[t]] == MiscDirection.WAIT:
        artists.append(
            ax.scatter(*xy[0], c='black', zorder=15, s=10)
        )

    ax.set_axis_off()

    # plot the vitals
    ax = fig.add_subplot(gs[0, 2])
    artists.extend(plot_series(
        blstats[:, NLE_BL_HP], title='HP', xlim=xlim, ax=ax,
        ylim=(0, blstats[:, NLE_BL_HPMAX].max()+1),
    ))

    # the in-game time-deltas
    ax = fig.add_subplot(gs[1, 2], sharex=ax)
    artists.extend(plot_series(
        np.ediff1d(blstats[:, NLE_BL_TIME], to_begin=0),
        title='Timedelta', xlim=xlim, ax=ax
    ))

    # the in-game score
    ax = fig.add_subplot(gs[2, 0], sharex=ax)
    artists.extend(plot_series(
        blstats[:, NLE_BL_SCORE], title='Game Score', xlim=xlim, ax=ax
    ))

    # the value estimates and the policy entropy
    ax = fig.add_subplot(gs[2, 1], sharex=ax)
    artists.extend(plot_series(
        ep.output.val['ext'], ep.output.val['int'],
        title='State Value Ext', xlim=xlim, ax=ax, ylim=None,
    ))
    artists.extend(plot_series(
        -(ep.output.pol * np.exp(ep.output.pol)).sum(-1),
        xlim=xlim, ax=ax.twinx(), c='C2',
        ylim=(0, math.log(len(proba))),
    ))

    # plot the rewards
    ax = fig.add_subplot(gs[2, 2], sharex=ax)
    artists.extend(plot_series(
        rew_.cumsum(), title='Reward', xlim=xlim, ax=ax,
    ))

    for ax in fig.axes:
        if ax is not view:
            artists.append(ax.axvline(t, c='r', zorder=+10))

    return artists
