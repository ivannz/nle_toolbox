import os
import json
import pickle

import numpy
import matplotlib.pyplot as plt

import gym
import nle

from joblib import delayed, Parallel
from hashlib import sha256

from .wrapper import Replay
from ..env.obs import BLStats


"""The series we compute for each trace."""
SERIES = {
    'reward__total':
        lambda rew, bls: float(
            rew.sum()
        ),

    'reward__sparsity':
        lambda rew, bls: float(
            (rew == 0).mean()
        ),

    'blstats__hitpoints__mean':
        lambda rew, bls: float(
            bls.hitpoints.mean()
        ),

    'blstats__hitpoints__median':
        lambda rew, bls: float(
            numpy.median(bls.hitpoints)
        ),

    'blstats__hitpoints__std':
        lambda rew, bls: float(
            bls.hitpoints.std()
        ),

    'blstats__experience_points__max':
        lambda rew, bls: float(
            bls.experience_points.max()
        ),

    'blstats__level_number__max':
        lambda rew, bls: float(
            bls.level_number.max()
        ),

    'blstats__energy__mean':
        lambda rew, bls: float(
            bls.energy.mean()
        ),

    'blstats__energy__std':
        lambda rew, bls: float(
            bls.energy.std()
        ),

    'blstats__gold__total':
        lambda rew, bls: float(
            bls.gold.sum()
        ),

    'blstats__score__max':
        lambda rew, bls: float(
            bls.score.max()
        ),

    'blstats__score__per_time':
        lambda rew, bls: float(
            bls.score.max() / bls.time.max()
        ),

    # frequency of various hunger states
    'blstats__hunger_state__counts':
        lambda rew, bls: numpy.bincount(
            bls.hunger_state,
            minlength=7,
        ).tolist(),
    }


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


def read_one(trace):
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
    return info, (
        numpy.array(rew),
        blstats,
    )


def evaluate_one(sh, trace, state_dict):
    """Compute the stats of the replayed episode."""

    try:
        info, (rewards, blstats, *ignored) = simulate_one(state_dict)

    except ValueError:
        return sh, dict(trace=trace, data={})

    # compute the statistics and supplement with the final infodict
    stats = {k: fn(rewards, blstats) for k, fn in SERIES.items()}
    return sh, dict(
        trace=trace,
        data={
            **stats, **{'info__' + k: v for k, v in info.items()},
        },
    )


def evaluate(folder, n_jobs=-1, verbose=10):
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
            continue

        # make sure the data contains at least the series specified at the top
        data = cache[sh]['data']
        if data and (SERIES.keys() - data.keys()):
            unready[sh] = trace, state_dict

    if unready:
        # dump the newly computed stats into a jayson file
        cache.update(Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_one)(k, *v) for k, v in unready.items()
        ))

        json.dump(cache, open(filename, 'wt'), indent=2)

    return cache


def series(cache, *names):
    """Extract the specified data from the cache."""
    names = list(filter(None, names))

    data = []
    for rec in cache.values():
        trace, stats = rec['trace'], rec['data']
        if not all(n in stats for n in names):
            continue

        data.append((trace, *(stats[n] for n in names)))

    traces, *data = zip(*data)
    return traces, numpy.array(data)


def main(
    folder,
    name='blstats__score__max',
    q=0.1,
    aux=None,
    *,
    plot=False,
    debug=False,
):
    breakpoint() if debug else None

    # get the specified series
    cache = evaluate(folder)

    try:
        traces, (main, *rest) = series(cache, name, aux)

    except ValueError:
        print(f"\n\nNo data for `{name}` in `{folder}`. Aborting...\n\n")
        return []

    # get the threshold and display a histogram
    lo = numpy.quantile(main, q=q)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=160)
        if aux:
            yaxis, *ignored = rest
            ax.set_title('Scatter plot')
            ax.scatter(main, yaxis, alpha=0.75)
            ax.set_ylabel(aux)

        else:
            ax.set_title(f"Histogram for `{name}`")
            ax.hist(main, bins=51, alpha=0.75)

        ax.set_xlabel(name)
        ax.axvline(lo, c='k', zorder=10)
        fig.tight_layout()
        fig.show()

        print("\n\nClick anywhere on the plot to continue...\n\n")
        plt.ginput()
        plt.close()

    return [
        (sc, tr) for tr, sc in zip(traces, main) if sc <= lo
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
        '--aux', type=str, default=None, required=False,
        help='The auxiliary series on the y-axis of a scatter plot.')

    parser.add_argument(
        '--q', type=float, default=0.1, required=False,
        help='The quantile [0, 1] for the most underperforming.')

    parser.add_argument(
        '--debug', required=False, dest='debug', action='store_true',
        help='Enter trace mode.')

    parser.set_defaults(
        q=0.1,
        name='blstats__score__max',
        aux=None,
        debug=False,
    )

    args, _ = parser.parse_known_args()
    pp.pprint(main(**vars(args), plot=True))
    print('\n\n')
