import numpy
import pytest

import gym
import nle

from random import choices

from nle_toolbox.utils.play.wrapper import Replay


def replay(env, actions, *, seed):
    history, obs_, gen = [], None, env.replay(actions, seed=seed)
    try:
        while True:
            obs, act, rew, obs_, info = next(gen)
            history.append(obs)

    except StopIteration as e:
        remaining = e.value

    # write the last observation, unless nothing has been written at all.
    if history:
        history.append(obs_)

    return history, remaining


def collate(records):
    return {
        k: numpy.array([
            rec[k] for rec in records
        ]) for k in records[0]
    }


@pytest.mark.parametrize(
    "k", [1, 10, 50, 100, 10000],
)
def test_utils_seeding_set_seed(k):
    # the first 20 actions are 8+8 navigation, up/down, wait, and next
    actions = choices(range(16), k=k)

    with Replay(gym.make("NetHackChallenge-v0")) as env:
        history, remaining1 = replay(env, actions, seed=None)
        obs_trace1 = collate(history)

        state_dict = env.state_dict()

    ref = history[-1]
    with Replay(gym.make("NetHackChallenge-v0")) as env:
        obs = env.load_state_dict(state_dict)

    assert all(
        (obs[k] == ref[k]).all() for k in obs.keys() | ref.keys()
    )

    with Replay(gym.make("NetHackChallenge-v0")) as env:
        history, remaining2 = replay(
            env,
            state_dict['actions'],
            seed=state_dict['seed'],
        )
        obs_trace2 = collate(history)

    # check the traces
    assert remaining1 == remaining2
    assert obs_trace1.keys() == obs_trace2.keys()

    assert all(
        (obs_trace1[k] == obs_trace2[k]).all()
        for k in obs_trace1.keys() | obs_trace2.keys()
    )
