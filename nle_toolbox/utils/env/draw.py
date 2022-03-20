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


def draw(fig, npy, t, *, actions, artists=None):
    artists = list() if artists is None else artists
    gs = GridSpec(3, 3)

    # the adjusted duration of the episode and the actions taken in the env
    n_length = len(npy.input.fin) - int(npy.input.fin[-1])
    act_ = plyr.apply(lambda x: x[1:], npy.input.act)  # XXX act[t] is a_{t-1}

    ep = plyr.apply(lambda x: x[:n_length], npy)
    ep_t = plyr.apply(lambda x: x[t], ep)

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

    ax = fig.add_subplot(gs[1:2, 2:], sharex=ax)
    ax.set_title('Timedelta')
    artists.extend(ax.plot(np.ediff1d(blstats[:, NLE_BL_TIME], to_begin=0)))
    artists.append(ax.axvline(t, c='r', zorder=+10))

    ax = fig.add_subplot(gs[2:3, :1], sharex=ax)
    ax.set_title('Game Score')
    artists.extend(ax.plot(blstats[:, NLE_BL_SCORE]))
    artists.append(ax.axvline(t, c='r', zorder=+10))

    ax = fig.add_subplot(gs[2:3, 1:2], sharex=ax)
    ax.set_title('LVL/XP')
    artists.extend(ax.plot(blstats[:, NLE_BL_EXP]))
    artists.extend(ax.plot(blstats[:, NLE_BL_XP]))
    artists.append(ax.axvline(t, c='r', zorder=+10))

    # plot the values and the entropy
    ax = fig.add_subplot(gs[2:3, 2:], sharex=ax)
    ax.set_title('State Value')
    artists.extend(ax.plot(ep.output.val['ext'], c='C0'))
    ent = -(ep.output.pol * np.exp(ep.output.pol)).sum(-1)
    artists.append(ax.axvline(t, c='r', zorder=+10))
    ax_ = ax.twinx()
    ax_.set_ylim(0., math.log(len(proba)))
    artists.extend(ax_.plot(ent, c='C1'))

    return artists
