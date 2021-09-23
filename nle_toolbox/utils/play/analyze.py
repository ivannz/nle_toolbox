import os
import json
import pickle

import numpy
import matplotlib.pyplot as plt

import gym
import nle

from joblib import delayed, Parallel
from hashlib import sha256

from ...wrappers.replay import Replay
from ..obs import BLStats


def traces(path, ext='.pkl'):
    """Get all replays in the specified folder."""

    path, _, snapshots = next(os.walk(os.path.abspath(path)))
    for basename, extension in map(os.path.splitext, snapshots):
        if extension != ext:
            continue

        filename = os.path.join(path, basename + ext)
        if os.stat(filename).st_size < 1:
            continue

        yield filename


def get_hash(seed, actions, **ignored):
    """Get a replay-compatible hash (file-based hashing is bad)."""

    sh = sha256()
    sh.update(pickle.dumps((seed, actions)))
    return sh.hexdigest()


def read_one(trace, *, _chunk_size=262144):
    """Read a replay file and get its hash."""

    state_dict = pickle.load(open(trace, 'rb'))
    return trace, get_hash(**state_dict), state_dict


def simulate_one(state_dict):
    """Replay the game and accumulate raw data."""

    data = []
    # XXX no options are overridden here!
    with Replay(gym.make('NetHackChallenge-v0')) as env:
        gen = env.replay(
            state_dict['actions'],
            seed=state_dict['seed'],
        )
        for obs, act, rew, obs_, info in gen:
            data.append((
                rew,
                tuple(obs['blstats'].tolist()),
            ))

    # collate stats
    rew, blstats, *ignored = zip(*data)
    blstats = BLStats(*numpy.array(blstats).T)

    # return the info at termination, and the collated stats
    return info, (numpy.array(rew), blstats,)


def evaluate_one(sh, trace, state_dict):
    """Compute the stats of the replayed episode."""

    info, (rewards, blstats, *ignored) = simulate_one(state_dict)

    # compute the statistics
    numeric = {
        'reward__total': rewards.sum(),
        'reward__sparsity': (rewards == 0).mean(),
        'blstats__hitpoints__mean': blstats.hitpoints.mean(),
        'blstats__hitpoints__std': blstats.hitpoints.std(),
        'blstats__energy__mean': blstats.energy.mean(),
        'blstats__energy__std': blstats.energy.std(),
        'blstats__gold__total': blstats.gold.sum(),
        'blstats__score__max': blstats.score.max(),
    }

    # frequency of various hunder states
    hunger_state = numpy.bincount(blstats.hunger_state, minlength=7).tolist()
    return sh, (
        trace, {
            'blstats__hunger_state__counts': hunger_state,
            **{k: float(v) for k, v in numeric.items()},
            **{'info__' + k: v for k, v in info.items()},
        },
    )


def evaluate(folder, *, n_jobs=-1, verbose=10):
    """Scan the specified folder for replay traces and compute missing stats.
    """

    folder = os.path.abspath(folder)

    # open the jayson metadata
    cache, filename = {}, os.path.join(folder, 'cache.json')
    if os.path.isfile(filename):
        cache = json.load(open(filename, 'rt'))

    unready = {}
    for trace, sh, state_dict in map(read_one, traces(folder)):
        # recompute the cache and the stats
        if sh not in cache:
            unready[sh] = trace, state_dict

    if unready:
        # dump the newly computed stats into a jayson file
        cache.update(Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_one)(k, *v) for k, v in unready.items()
        ))

        json.dump(cache, open(filename, 'wt'), indent=2)

    return cache


def series(cache, name='blstats__score__max'):
    """Extract the specified data from the cache."""

    data = []
    for sh, (trace, stats) in cache.items():
        data.append((
            trace, stats[name]
        ))

    traces, data = zip(*data)
    return traces, numpy.array(data)


def main(
    folder,
    name='blstats__score__max',
    q=0.1,
    *,
    plot=False,
    debug=False,
):
    breakpoint() if debug else None

    # get the specified series
    cache = evaluate(folder)
    traces, scores = series(cache, name)

    # get the threshold and display a histogram
    lo = numpy.quantile(scores, q=q)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=160)
        ax.set_title(f"Histogram for `{name}`")
        ax.hist(scores, bins=51, alpha=0.75)
        ax.axvline(lo, c='k', zorder=10)
        fig.show()

        print("\n\nClick anywhere on the plot to continue...\n\n")
        plt.ginput()
        plt.close()

    return [
        tr for tr, sc in zip(traces, scores) if sc <= lo
    ]


if __name__ == '__main__':
    import argparse
    import pprint as pp

    parser = argparse.ArgumentParser(
        description='Interactively replay a recorded playthrough.',
        add_help=True)

    parser.add_argument(
        'folder', type=str,
        help='The stored replay data.')

    parser.add_argument(
        '--name', type=str, default='blstats__score__max', required=False,
        help='The score series to analyze.')

    parser.add_argument(
        '--q', type=float, default=0.1, required=False,
        help='The quantile [0, 1] for the most underperforming.')

    parser.add_argument(
        '--debug', required=False, dest='debug', action='store_true',
        help='Enter trace mode.')

    parser.set_defaults(q=0.1, name='blstats__score__max', debug=False)

    args, _ = parser.parse_known_args()
    pp.pprint(main(**vars(args), plot=True))
    print('\n\n')
