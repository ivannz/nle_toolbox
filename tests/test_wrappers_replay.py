import numpy
import pytest

import gym
import nle

from random import choices

from nle_toolbox.wrappers.replay import Replay


@pytest.mark.parametrize(
    "k", [1, 10, 50, 100, 10000],
)
def test_utils_seeding_set_seed(k):
    # the first 20 actions are 8+8 navigation, up/down, wait, and next
    actions = choices(range(16), k=k)

    with Replay(gym.make("NetHackChallenge-v0")) as env:
        history, remaining1 = env.replay(actions, seed=None)
        state_dict = env.state_dict()

        obs_trace1 = {
            k: numpy.array([o[k] for o in history]) for k in history[0]
        }

    with Replay(gym.make("NetHackChallenge-v0")) as env:
        history, remaining2 = env.load_state_dict(state_dict)

        obs_trace2 = {
            k: numpy.array([o[k] for o in history]) for k in history[0]
        }

    # check the traces
    assert remaining1 == remaining2
    assert obs_trace1.keys() == obs_trace2.keys()

    assert all(
        (obs_trace1[k] == obs_trace2[k]).all()
        for k in obs_trace1.keys() | obs_trace2.keys()
    )
