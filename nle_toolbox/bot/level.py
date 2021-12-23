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
from ..utils.env.obs import npy_fold2d


dt_map = np.dtype([
    # the coordinates' backreference
    ('rc', np.dtype([('r', int), ('c', int)])),
    # the glyph id
    ('glyph', int),
    # visitation counter
    ('n_visited', int),
    # the number of times this xy was updated
    ('n_updates', int),
    ('n_last_updated', int),
    # area it belongs to
    ('area', int),
    # foreground object flag
    # ('is_foreground', bool),  # equals `not .info.is_background`
    # pathfinding cost
    ('cost', float),
    # the extracted glyph info
    ('info', dt_glyph_ext),
])


class Level:
    def __init__(self, shape=DUNGEON_SHAPE, k=1):
        # the data is bordered array, for easier folding for
        #  the vicinity viewport. `glyph` determines if it
        #  is in the valid map region.
        rows, cols = shape
        data = np.empty((2, k + rows + k, k + cols + k), dtype=dt_map)

        # fill with default values for the border
        data[:] = (
            (-1, -1),    # invalid row-col coords
            MAX_GLYPH,   # invalid glyph
            0,           # was never visited
            0,           # never got updated
            -1,          # last updated info is not available
            -1,          # parent flood-fill area is the Universum
            # False,       # is a foreground glyph
            float('inf'),  # prohibitively expensive for pathfinding
            ext_glyphlut[MAX_GLYPH],  # extra info for the invalid gylph
        )

        # fill in the row-col coordinate for backreference
        tiles = self.bg_tiles, self.stg_tiles, \
            = data[:, k:-k, k:-k].view(np.recarray)
        for r, c in np.ndindex(rows, cols):
            tiles.rc[:, r, c] = r, c

        # setup read-only view for adjacent tiles
        self.bg_vicinity, self.stg_vicinity, \
            = npy_fold2d(
                data, k=k, n_leading=1, writeable=False,
            ).view(np.recarray)

        # sparse data structures with row-col keys (unbordered coords)
        # sparsely populated dict keyed by row-col containing arbitrary data,
        # e.g. item piles, the monster population, special designations etc.
        self.data = {}
        # XXX we do not use defdict here, since it spawns defaults on
        # read-access, which beats its utility

        # the trace of the row-col coords from bls through the level
        self.trace = [(-1, -1)]

        # the total number of map updates so far
        self.n_updates = 0

    def update(self, obs):
        """Take the currently observed glyphs and update our representation.
        """
        glyphs = obs['glyphs']

        # differentially update the glyph staging layer
        stage = self.stg_tiles
        diff = glyphs != stage.glyph
        # diff |= (stage.n_last_updated < self.n_updates - 512)

        n_diff = diff.sum()
        if n_diff < 1:
            return

        # the game is not in gui mode and the player is not engulfed
        self.n_updates += 1

        # update glyphs, the step number, the number of updates, and fetch
        #  new metadata
        glyphs = stage.glyph[diff] = glyphs[diff]
        stage.n_last_updated[diff] = self.n_updates
        stage.n_updates[diff] += 1

        stage.info[diff] = ext_glyphlut[glyphs]

        # trace the player's coordinates through the level
        bls = obs['blstats']
        r, c = location = bls[NLE_BL_Y], bls[NLE_BL_X]
        if self.trace[-1] != location:
            stage.n_visited[location] += 1
            self.trace.append(location)

        # update bg only if the __staged background__ glyph is NOT the same
        #  as the current bg glyph. This way we permit a certain degree of
        #  stickyness to bg.
        to_bg = (stage.glyph != self.bg_tiles.glyph) & stage.info.is_background
        self.bg_tiles[to_bg] = stage[to_bg]

        # enumerate the list of `interesting` objects
        # `is_accessible` -- used for pathfinding to determine the base cost
        # `is_actor`, `is_trap` --
        # stage.info.is_interesting


class DungeonMapper:
    def __init__(self):
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
        self.vw_window = npy_fold2d(data, k=1, n_leading=0, writeable=False)

        # track the path through the dungeons
        self.trace = [(-1, -1)]

        # the current dungeon level
        self.level = None

    def update(self, obs):
        # update the dungeon's internal buffers
        np.copyto(self.vw_glyphs, obs['glyphs'], 'same_kind')
        bls = obs['blstats']

        # we need to handle being swallowed separately
        window = self.vw_window[bls[NLE_BL_Y], bls[NLE_BL_X]]
        self.is_swallowed = any(map(glyph_is.swallow, window.flat))
        # XXX when engulfed the game clears the glyphs, i.e. replaces
        #  the map with `S_stone`, and draw the insides of the monster
        #  around the current location of the player.
        #    c.f. [`swallowed`](./nle/src/display.c#L1115-1179)

        # update the trace through the dungeons
        # XXX [`on_level`](./nle/src/dungeon.c#L1095-1102) compares levels
        level = bls[NLE_BL_DNUM], bls[NLE_BL_DLEVEL]
        if self.trace[-1] != level:
            self.trace.append(level)

        # update the level, unless swallowed
        if not self.is_swallowed:
            if level not in self.maps:
                self.maps[level] = Level(shape=DUNGEON_SHAPE, k=1)

            self.maps[level].update(obs)

        self.level = self.maps.get(self.trace[-1])
