import numpy
import pytest

import gym
import nle

from random import Random, randint

from nle_toolbox.utils.seeding import set_seed


def replay(env, actions, *, seed):
    set_seed(env, seed=seed)

    history = [env.reset()]
    for j, act in enumerate(actions):
        obs, rew, fin, info = env.step(act)
        history.append(obs)
        if fin:
            break

    return {
        k: numpy.array([o[k] for o in history]) for k in history[0]
    }, actions[j:]


@pytest.mark.parametrize(
    "seed,k", [
        (randint(1, 7857458), 125) for _ in range(100)
    ],
)
def test_utils_seeding_set_seed(seed, k):

    # get out own prng for actions
    mt19937 = Random(seed)
    nle_seed = mt19937.getrandbits(128)

    # the first 20 actions are 8+8 navigation, up/down, wait, and next
    actions = mt19937.choices(range(16), k=k)

    # Does init seed affect anything at all?
    with gym.make('NetHackChallenge-v0') as env:
        obs_trace1, remaining1 = replay(env, actions, seed=nle_seed)

    # make sure not to reuse the same env twice
    with gym.make('NetHackChallenge-v0') as env:
        obs_trace2, remaining2 = replay(env, actions, seed=nle_seed)

    # check the traces
    assert remaining1 == remaining2
    assert obs_trace1.keys() == obs_trace2.keys()

    assert all(
        (obs_trace1[k] == obs_trace2[k]).all()
        for k in obs_trace1.keys()
    )
