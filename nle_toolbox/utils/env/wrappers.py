import numpy as np
import gym

from collections import deque

from .render import fixup_tty
from ..fold import npy_fold2d

from nle.nethack import (
    MAX_GLYPH,
    NLE_BL_X,
    NLE_BL_Y,
    NLE_BL_STR25,
    NLE_BL_STR125,
    DUNGEON_SHAPE,
)


# from gym import ObservationWrapper, ActionWrapper
class ObservationWrapper(gym.ObservationWrapper):
    """Fixup the somewhat heavy handed `.reset` patch in `ObservationWrapper`.
    """
    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))


class ActionWrapper(gym.ActionWrapper):
    """Fixup the somewhat heavy handed `.reset` patch in `ActionWrapper`.
    """
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NLEAtoN(ActionWrapper):
    """Map ascii characters into NLE's actions.

    Details
    -------
    Allowing the original ascii characters instead of the opaque action
    integers enables the design and architecture building process simpler,
    and more attached to the swathes of documentation on NetHack.
    """
    from nle.nethack import ACTIONS

    def __init__(self, env):
        super().__init__(env)
        # XXX for `NetHackChallenge` we could rely on `ACTIONS`
        #  see ./nle/env/tasks.py#L52,328
        self.ctoa = {
            chr(a): j for j, a in enumerate(self.unwrapped.actions)
        }

    def action(self, action):
        return self.ctoa[action]


class NLEObservationPatches(ObservationWrapper):
    """Patch the tty character data."""
    def observation(self, observation):
        # apply tty rendering patches
        if 'tty_chars' in observation:
            # XXX assume we-ve got all `tty_*` stuff
            observation.update(fixup_tty(**observation))

        return observation


class NLEFeatureExtractor(ObservationWrapper):
    """Features for the neural controller.

    Vicinity
    --------
    This is an ego-centric view of the specified radius (`k`) into the dungeon
    map.

    Strength
    --------
    The strength stat in NetHack, which is based on AD&D 2ed mechanics, comes
    in two ints: strength and percentage strength. The latter is applicable to
    **warrior classes** with **natural strength 18** only and denotes
    `exceptional strength`, which confers extra chance-to-hit, increased
    damage, and higher chance to force locks or doors.
    """
    def __init__(self, env, *, k=3):
        super().__init__(env)

        # we extend the observation space
        decl = self.observation_space['glyphs']

        # allocate a bordered array for glyphs
        rows, cols = DUNGEON_SHAPE
        glyphs = self.glyphs = np.full((
            k + rows + k, k + cols + k,
        ), MAX_GLYPH, dtype=decl.dtype)

        # create view for fast access
        self.vw_glyphs = glyphs[k:-k, k:-k]
        self.vw_vicinity = npy_fold2d(
            glyphs, k=k, n_leading=0, writeable=True,
            # XXX torch does not like read-only views
        )

        # declare `vicinity` in the observation space
        self.observation_space['vicinity'] = gym.spaces.Box(
            0, MAX_GLYPH,
            dtype=self.vw_vicinity.dtype,
            shape=self.vw_vicinity.shape[2:],
        )

    def observation(self, observation):
        # recv a new observation
        blstats = observation['blstats'].copy()

        # copy glyphs into the glyph area and then extract the vicinity at X, Y
        np.copyto(self.vw_glyphs, observation['glyphs'], 'same_kind')
        vicinity = self.vw_vicinity[blstats[NLE_BL_Y], blstats[NLE_BL_X]]

        # strength percentage is more detailed than `str` stat
        # XXX compare src/winrl.cc#L538 with src/attrib.c#L1072-1085
        #     e.g. src/dokick.c#L38 sums the transformed str with dex and con
        str, prc = blstats[NLE_BL_STR125], 0.
        if str >= 122:
            str = min(str - 100, 25)

        elif str >= 19:
            str, prc = divmod(19 + str / 50, 1)  # divmod-by-one :)

        blstats[NLE_BL_STR25] = int(str)
        blstats[NLE_BL_STR125] = int(prc * 100)  # original step .02, so ok

        # update the observation dict inplace
        observation.update(dict(blstats=blstats, vicinity=vicinity.copy()))
        return observation


class ObservationDictFilter(ObservationWrapper):
    """Allow the specified fields in the observation dict.
    """
    def __init__(self, env, *keys):
        super().__init__(env)
        self.keys = frozenset(keys)

        self.observation_space = gym.spaces.Dict(
            self.observation(self.observation_space)
        )

    def observation(self, observation):
        return {k: v for k, v in observation.items() if k in self.keys}


class RecentHistory(gym.Wrapper):
    """The base interaction architecture is essentially a middleman, who passes
    the action to the underlying env and intercepts the resulting transition
    data. It also is allowed, but not obliged to interact with the env, while
    intercepting the observations.
    """
    def __new__(cls, env, *, n_recent=0, map=None):
        if n_recent < 1:
            return env
        return object.__new__(cls)

    def __init__(self, env, *, n_recent=0, map=None):
        super().__init__(env)
        self.recent = deque([], n_recent)
        self.map = map if callable(map) else lambda x: x

    def reset(self, seed=None):
        self.recent.clear()
        return self.env.reset()

    def step(self, action):
        self.recent.append(self.map(action))
        return self.env.step(action)
