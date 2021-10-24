import numpy as np

from gym import ObservationWrapper, ActionWrapper

from ..utils.env.obs import fold2d, BLStats
from ..utils.env.render import fixup_tty
from ..utils.env.defs import special

from nle.nethack import NLE_BL_STR25, NLE_BL_STR125


class NLEAtoN(ActionWrapper):
    """Map ascii characters into NLE's actions."""
    from nle.nethack import ACTIONS

    def __init__(self, env):
        super().__init__(env)
        # XXX for `NetHackChallenge` we could rely on `ACTIONS`
        #  see ./nle/env/tasks.py#L52,328
        self.ctoa = {
            chr(a): j for j, a in enumerate(self.unwrapped._actions)
        }

    def action(self, action):
        return self.ctoa[action]


class NLEPatches(ObservationWrapper):
    """Patch the tty character data."""
    def __init__(self, env, *, copy=False):
        super().__init__(env)
        self.copy = copy

    def observation(self, observation):
        # make a copy
        if self.copy:
            observation = {k: a.copy() for k, a in observation.items()}

        # apply tty rendering patches
        if 'tty_chars' in observation:
            # XXX assume we-ve got all `tty_*` stuff
            observation.update(fixup_tty(**observation))

        return observation


class NLEFeatures(ObservationWrapper):
    """Features for the neural controller."""
    def __init__(self, env, *, k=3):
        super().__init__(env)

        from nle.nethack import DUNGEON_SHAPE, MAX_GLYPH, NLE_INVENTORY_SIZE

        # create bordered glyph array
        rows, cols = DUNGEON_SHAPE
        glyphs = self.glyphs = np.full((
            k + rows + k, k + cols + k,
        ), MAX_GLYPH, dtype=int)  # silently promote from int16 to int64

        self.inv_glyphs = np.full(NLE_INVENTORY_SIZE, MAX_GLYPH, dtype=int)

        # create view for fas access
        self.vw_glyphs = glyphs[k:-k, k:-k]
        self.vw_vicinity = fold2d(glyphs, k=k, leading=0, writeable=False)

    def observation(self, observation):
        # strength percentage is more detailed than `str` stat
        # XXX compare src/winrl.cc#L538 with src/attrib.c#L1072-1085
        #     e.g. src/dokick.c#L38 sums the transformed str with dex and con
        blstats = observation['blstats']
        str, prc = blstats[NLE_BL_STR125], 0.
        if str >= 122:
            str = min(str - 100, 25)

        elif str >= 19:
            str, prc = divmod(19 + str / 50, 1)  # divmod-by-one :)
        blstats[NLE_BL_STR25] = int(str)
        blstats[NLE_BL_STR125] = int(prc * 100)  # original step .02, so ok

        # recv new observation
        bls = BLStats(*observation['blstats'])
        np.copyto(self.vw_glyphs, observation['glyphs'], 'same_kind')
        np.copyto(self.inv_glyphs, observation['inv_glyphs'], 'same_kind')

        is_objpile = observation['specials'] & special.OBJPILE

        observation.update(dict(
            glyphs=self.vw_glyphs,
            inv_glyphs=self.inv_glyphs,
            blstats=bls,
            vicinity=self.vw_vicinity[bls.y, bls.x],
            is_objpile=is_objpile.astype(bool),
        ))
        return observation
