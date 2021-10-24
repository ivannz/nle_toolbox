import numpy as np
from nle.nethack import (
    NLE_BL_X,
    NLE_BL_Y,
    NLE_BL_DNUM,
    NLE_BL_DLEVEL,
    # NLE_BL_DEPTH,  # derived from DNUM and DLEVEL
    # XXX does not uniquely identify floors,
    #  c.f. [`depth`](./nle/src/dungeon.c#L1086-1084)
    DUNGEON_SHAPE,
    MAX_GLYPH,
)

from .chassis import InteractiveWrapper, Chassis, get_wrapper

from ..utils.env.defs import glyph_is, dt_glyph_ext, ext_glyphlut
from ..utils.env.obs import fold2d


dt_map = np.dtype([
    # the coordinates' backreference
    ('xy', np.dtype([('x', int), ('y', int)])),
    # the glyph id
    ('glyph', int),
    # boolean visited flag
    ('n_visited', int),
    # the number of times this xy was updated
    ('n_updates', int),
    ('n_last_updated', int),
    # area it belongs to
    ('area', int),
    # the extracted glyph info
    ('info', dt_glyph_ext),
])


class Level:
    def __init__(self, shape=DUNGEON_SHAPE, k=1):
        # the data is bordered array, for easier folding for
        #  the vicinity viewport. `glyph` determines if it
        #  is in the valid map region.
        rows, cols = shape
        data = np.empty((3, k + rows + k, k + cols + k), dtype=dt_map)

        # fill with default values for the border
        data[:] = (
            (-1, -1),    # invalid x-y coords
            MAX_GLYPH,   # invalid glyph
            0,           # was never visited
            0,           # never got updated
            -1,          # last updated info is not available
            -1,          # parent flood-fill area is the Universum
            ext_glyphlut[MAX_GLYPH],  # extra info for the invalid gylph
        )

        # fill in the x-y coordinate for backreference
        tiles = self.bg_tiles, self.fg_tiles, self.stg_tiles, \
            = data[:, k:-k, k:-k].view(np.recarray)
        for x, y in np.ndindex(rows, cols):
            tiles.xy[:, x, y] = x, y

        # setup read-only view for adjacent tiles
        self.bg_vicinity, self.fg_vicinity, self.stg_vicinity, \
            = fold2d(data, k=k, leading=1, writeable=False).view(np.recarray)

        # sparse data structures with x-y keys (unbordered coords)
        # sparsely populated dict keyed by x-y containing arbitrary data, e.g.
        # item piles, the monster population, special designations etc.
        self.data = {}
        # XXX we do not use defdict here, since it spawns defaults on
        # read-access, which beats its utility

        # the trace of the x-y coords from bls through the level
        self.trace = [(-1, -1)]

        # the total number of map updates so far
        self.n_updates = 0

    def update(self, obs):
        """Take the currently observed glyphs and update our representation.
        """
        glyphs = obs['glyphs']

        # the game is not in gui mode and the player is not engulfed
        self.n_updates += 1

        # differentially update the glyph staging layer
        stage = self.stg_tiles
        diff = glyphs != stage.glyph
        # diff |= (stage.n_last_updated < self.n_updates - 512)

        n_diff = diff.sum()
        if n_diff < 1:
            return

        # save previous data and update glyphs
        old = stage[diff]
        new = stage.glyph[diff] = glyphs[diff]

        # update the last time, the number of updates, and properties metadata
        stage.n_updates[diff] += 1
        stage.n_last_updated[diff] = self.n_updates
        stage.info[diff] = ext_glyphlut[new]

        # trace the player's coordinates through the level
        bls = obs['blstats']
        r, c = location = bls[NLE_BL_Y], bls[NLE_BL_X]
        if self.trace[-1] != location:
            stage.n_visited[location] += 1
            self.trace.append(location)

        # 3. analyze the staged glyphs and split into bg/fg
        # xy, glyph, n_visited, n_updates, n_last_updated, area, info
        pass


class DungeonMapper(InteractiveWrapper):
    def __init__(self, env):
        super().__init__(env)

        # we need a reference to the underlying chassis wrapper
        self.chassis = get_wrapper(env, Chassis)

        # NetHack identifies the dungeons and levels with 'level_number' and
        #  'dungeon_number'. The 'depth' is computed from these values in
        #  [`depth`](./nle/src/dungeon.c#L1086-1084), and ultimately depends on
        #  the loading order in [`init_dungeons`](./nle/src/dungeon.c#L712-978)
        # XXX The comment in dungeon.c says:
        #  >>> ... levels in different dungeons can have the same depth.
        self.maps = {(-1, -1): Level(shape=(3, 3), k=1)}

        # the intermediate glyph buffer and 3x3 window
        rows, cols = DUNGEON_SHAPE
        data = np.full((1 + rows + 1, 1 + cols + 1), MAX_GLYPH, dtype=int)

        self.vw_glyphs = data[1:-1, 1:-1]
        self.vw_window = fold2d(data, k=1, leading=0, writeable=False)

        # track the path through the dungeons
        self.trace = [(-1, -1)]

        # the current dungeon level
        self.level = None

    def update(self, obs, rew=0., done=False, info=None):
        # update level representation unless we're in a menu,
        #  waiting for a prompt, or in the game over screen.
        if not (self.chassis.in_menu or self.chassis.prompt or done):
            bls = obs['blstats']

            # update the dungeon's internal buffers
            np.copyto(self.vw_glyphs, obs['glyphs'], 'same_kind')

            # we need to handle being swallowed separately
            window = self.vw_window[bls[NLE_BL_Y], bls[NLE_BL_X]]
            self.is_swallowed = any(map(glyph_is.swallow, window.flat))
            # XXX when engulfed the game clears the glyphs, i.e. replaces
            #  the map with `S_stone`, and draw the insides of the monster
            #  around the current location of the player.
            #    c.f. [`swallowed`](./nle/src/display.c#L1115-1179)

            # determine the dungeon level
            level = bls[NLE_BL_DNUM], bls[NLE_BL_DLEVEL]

            # update the trace through the dungeons
            # XXX [`on_level`](./nle/src/dungeon.c#L1095-1102) compares levels
            if self.trace[-1] != level:
                self.trace.append(level)

            # update the level, unless swallowed
            if not self.is_swallowed:
                if level not in self.maps:
                    self.maps[level] = Level()

                self.maps[level].update(obs)

            else:
                pass

            self.level = self.maps.get(self.trace[-1])

        return obs, rew, done, info
