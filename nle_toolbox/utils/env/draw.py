import math
import plyr

import numpy as np
from scipy.special import softmax

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

    # normalize and scale the directonal vectors
    C = sum(dirs.values())
    return {
        CompassDirection[k]:
        z * dirs[k] / C for k, z in compass_to_complex.items()
    }


def draw(fig, npy, t, *, actions, artists=None, view=None):
    artists = list() if artists is None else artists
    gs = GridSpec(3, 3)

    # the adjusted duration of the episode and the actions taken in the env
    n_length = len(npy.input.fin) - int(npy.input.fin[-1])
    act_ = plyr.apply(lambda x: x[1:], npy.input.act)  # XXX act[t] is a_{t-1}

    ep = plyr.apply(lambda x: x[:n_length], npy)
    ep_t = plyr.apply(lambda x: x[t], ep)

    # compute the moving view port around `t`
    if view is not None:
        if isinstance(view, float):
            view = max(1, int(n_length * view))

        if not isinstance(view, int):
            raise ValueError("`view` must be float or int.")

        # get the center of the "viewport" [x - v, x + v]
        x = min(max(t, view), n_length - 1 - view)
        view = x - view, x + view  # XXX make vw stay within [0, T-1]

    # the current policy and the botl stats
    proba = dict(zip(actions, softmax(ep_t.output.pol, axis=-1).tolist()))
    blstats = ep.input.obs['blstats']

    # render the agent's view with the compass action distribution overlay
    ax = fig.add_subplot(gs[:2, :2])
    ax.set_axis_off()
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

    # plot the vitals and time
    ax = fig.add_subplot(gs[0:1, 2:])
    ax.set_title('HP')
    artists.extend(ax.plot(blstats[:, NLE_BL_HP]))
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax.set_ylim(-1, blstats[:, NLE_BL_HPMAX].max() + 1)
    ax.yaxis.set_visible(False)  # the values are irrelevant
    if view is not None:
        ax.set_xlim(view)

    series = np.ediff1d(blstats[:, NLE_BL_TIME], to_begin=0)
    ax = fig.add_subplot(gs[1:2, 2:], sharex=ax)
    ax.set_title('Timedelta')
    artists.extend(ax.plot(series))
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax.yaxis.set_visible(False)
    if view is not None:
        ax.set_ylim(-1, series.max())
        ax.set_xlim(view)

    series = blstats[:, NLE_BL_SCORE]
    ax = fig.add_subplot(gs[2:3, :1], sharex=ax)
    ax.set_title('Game Score')
    artists.extend(ax.plot(series))
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax.yaxis.set_visible(False)
    if view is not None:
        ax.set_ylim(-1, series.max())
        ax.set_xlim(view)

    ax = fig.add_subplot(gs[2:3, 1:2], sharex=ax)
    ax.set_title('LVL/XP')
    artists.extend(ax.plot(blstats[:, NLE_BL_EXP]))
    artists.extend(ax.plot(blstats[:, NLE_BL_XP]))
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax.yaxis.set_visible(False)
    if view is not None:
        ax.set_xlim(view)

    # plot the values and the entropy
    ax = fig.add_subplot(gs[2:3, 2:], sharex=ax)
    ax.set_title('State Value')
    artists.extend(ax.plot(ep.output.val['ext'], c='C0'))
    artists.extend(ax.plot(ep.output.val['int'], c='C1'))
    ent = -(ep.output.pol * np.exp(ep.output.pol)).sum(-1)
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax.yaxis.set_visible(False)
    if view is not None:
        ax.set_xlim(view)

    ax_ = ax.twinx()
    ax_.set_ylim(0., math.log(len(proba)))
    artists.extend(ax_.plot(ent, c='C2'))

    return artists
