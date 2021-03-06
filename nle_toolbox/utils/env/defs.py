"""Various useful defs, macros and logic mined from NetHack and the NLE.
"""
# flake8: noqa
# pycodestyle: noqa
# ignore pep violations since this is mostly copied from c
#   E221 - multiple spaces before operator
#   E114 - indentation is not a multiple of four (comment)
#   E116 - unexpected indentation (comment)

import sys
import numpy as np

from collections import namedtuple

# bottom line indices
from nle.nethack import (
    NLE_BL_X,
    NLE_BL_Y,
    NLE_BL_STR25,
    NLE_BL_STR125,
    NLE_BL_DEX,
    NLE_BL_CON,
    NLE_BL_INT,
    NLE_BL_WIS,
    NLE_BL_CHA,
    NLE_BL_SCORE,
    NLE_BL_HP,
    NLE_BL_HPMAX,
    NLE_BL_DEPTH,
    NLE_BL_GOLD,
    NLE_BL_ENE,
    NLE_BL_ENEMAX,
    NLE_BL_AC,
    NLE_BL_HD,
    NLE_BL_XP,
    NLE_BL_EXP,
    NLE_BL_TIME,
    NLE_BL_HUNGER,
    NLE_BL_CAP,
    NLE_BL_DNUM,
    NLE_BL_DLEVEL,
    NLE_BL_CONDITION,
    NLE_BL_ALIGN,
)


# offsets
from nle.nethack import (
    GLYPH_MON_OFF,  # a monster
    GLYPH_PET_OFF,  # a pet
    GLYPH_INVIS_OFF,  # invisible
    GLYPH_DETECT_OFF,  # mon detect
    GLYPH_BODY_OFF,  # a corpse
    GLYPH_RIDDEN_OFF,  # mon ridden
    GLYPH_OBJ_OFF,  # object
    GLYPH_CMAP_OFF,  # cmap
    GLYPH_EXPLODE_OFF,  # explosion
    GLYPH_ZAP_OFF,  # zap beam
    GLYPH_SWALLOW_OFF,  # swallow
    GLYPH_WARNING_OFF,  # warn flash
    GLYPH_STATUE_OFF,  # a statue
    MAX_GLYPH,  # (end)
)

# sizes of different glyph groups
from nle.nethack import (
    NUMMONS,
    NUM_OBJECTS,
    MAXEXPCHARS,
    MAXPCHARS,
    EXPL_MAX,
    NUM_ZAP,
    WARNCOUNT,
)


# various constants
from nle.nethack import (
    MAXOCLASSES,
    MAXMCLASSES,
)


# the number of distinct entities that glyphs actually correspond to
MAX_ENTITY = (
    NUMMONS  # normal monsters
    + 0  # pets: mapped to normal monsters
    + 1  # single invisible monster placeholder
    + 0  # detected (?) monsters: mapped to normal
    + 0  # coprses: mapped to normal (not undead)
    + 0  # mounts: mapped to monsters
    + NUM_OBJECTS  # in-game items
    + (MAXPCHARS - MAXEXPCHARS)  # level topology, walls, corridors, doors etc.
    + EXPL_MAX  # explosion types: ignoring geometry
    + NUM_ZAP  # zap (spells projectiles): ignoring direction
    + 0  # monsters' insides: mapped to normal monsters
    + WARNCOUNT  # danger vibes from unknown monsters
    + 0  # monster statues: mapped to normal monsters
)


# /* hunger states - see hu_stat in src/eat.c#L74-76 */
class hunger:
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4
    FAINTED = 5
    STARVED = 6

    MAX = 7

    # /* see hunger states in hack.h#L44-53 - texts used on bottom line */
    _name = tuple(
        map(
            sys.intern,
            (
                "satiated",
                # we use a meaningful descr, instead of whitespace
                "not hungry",
                "hungry",
                "weak",
                "fainting",
                "fainted",
                "starved",
            ),
        )
    )


# /* Boolean condition bits for the condition mask */
# bitfields are copied from include/botl.h#L56-70
#  and groups from src/botl.c#L2321-2333
class condition:
    # major
    STONE = 0x00000001
    SLIME = 0x00000002
    STRNGL = 0x00000004
    FOODPOIS = 0x00000008
    TERMILL = 0x00000010
    MAJOR = STONE | SLIME | STRNGL | FOODPOIS | TERMILL

    # minor
    BLIND = 0x00000020
    DEAF = 0x00000040
    STUN = 0x00000080
    CONF = 0x00000100
    HALLU = 0x00000200
    MINOR = BLIND | DEAF | STUN | CONF | HALLU

    # movement
    LEV = 0x00000400
    FLY = 0x00000800
    RIDE = 0x00001000
    MOVEMENT = LEV | FLY | RIDE

    N_BITS = 13

    ALL = MAJOR | MINOR | MOVEMENT

    # /* see hunger states in hack.h#L44-53 - texts used on bottom line */
    _name = {
        STONE: "pterified",
        SLIME: "turned to slime",
        STRNGL: "strnagled",
        FOODPOIS: "food poisoning",
        TERMILL: "terminally ill",
        BLIND: "blind",
        DEAF: "deaf",
        STUN: "stunned",
        CONF: "confused",
        HALLU: "hallucianting",
        LEV: "levitating",
        FLY: "flying",
        RIDE: "riding",
    }


# /* symbolic names for capacity levels */
# taken from include/hack/h#L23-30
class encumberance:
    UNENCUMBERED = 0
    SLT_ENCUMBER = 1  # /* Burdened */

    MOD_ENCUMBER = 2  # /* Stressed */
    HVY_ENCUMBER = 3  # /* Strained */
    EXT_ENCUMBER = 4  # /* Overtaxed */

    OVERLOADED = 5  # /* Overloaded */

    MAX = 6

    _name = tuple(
        map(
            sys.intern,
            (
                "unencumbered",
                "burdened",
                "stressed",
                "strained",
                "overtaxed",
                "overloaded",
            ),
        )
    )


# /* Special returns from mapglyph() */
# from include/hack.h#L76-84
class special:
    """the bitfields of the `special` data in the observation from the NLE.

    Details
    -------
    These seem to be completely determined by the glyph, at least those,
    related to monsters. OBJPILE is interesting though.

        see src/maglyph.c#L60-254
    """

    # determined by the glyph
    from nle.nethack import MG_CORPSE as CORPSE  # 0x01
    from nle.nethack import MG_INVIS as INVIS  # 0x02
    from nle.nethack import MG_DETECT as DETECT  # 0x04
    from nle.nethack import MG_PET as PET  # 0x08
    from nle.nethack import MG_RIDDEN as RIDDEN  # 0x10
    from nle.nethack import MG_STATUE as STATUE  # 0x20

    # boulders hide objpiles, corpses and statues count towards objpiles
    # NB affected by hallucination src/maglyph.c#L57-58
    # /* more than one stack of objects */
    from nle.nethack import MG_OBJPILE as OBJPILE  # 0x40

    # determined by the glyph and and whether the color is being used and
    #  the water/lava symbols coincide.
    from nle.nethack import MG_BW_LAVA as BW_LAVA  # 0x80

    # /* 'black & white lava': highlight lava if it
    #    can't be distringuished from water by color */


# cmap symbol-glyph semantics
class symbol:
    """Copied verbatim and REORDERED from include/rm.h"""

    # /* begin dungeon characters */

    # begin IS_ROCK according to rm.h:89 and hack.c `test_move` (713), `bad_rock` (659)
    # XXX all are impassable, unless the hero is polymorphed or etheral,
    #  can tunnel or has a pick axe (digging implement). See also mondata.c and
    S_stone = 0
    S_tree = 18  # /* KMH */  # passable if hero has an axe, but not ethereal.
    # XXX Replaces stone in arboreal dungeons. Can also be a standalone tree.
    # begin walls
    S_vwall = 1
    S_hwall = 2
    S_tlcorn = 3
    S_trcorn = 4
    S_blcorn = 5
    S_brcorn = 6
    S_crwall = 7
    S_tuwall = 8
    S_tdwall = 9
    S_tlwall = 10
    S_trwall = 11
    # end walls

    # XXX also passable if hero can corrode, ooze or amorphous
    S_bars = 17  # /* KMH -- iron bars */

    # XXX also passable if hero can ooze or amorphous
    S_vcdoor = 15  # /* closed door, vertical wall */
    S_hcdoor = 16  # /* closed door, horizontal wall */

    # XXX passable if ethereal
    S_vcdbridge = 37  # /* closed drawbridge, vertical wall */
    S_hcdbridge = 38  # /* closed drawbridge, horizontal wall */

    # SDOOR  # a wall that is a secret door apply.c:434
    # SCORR  # a wall that is a secret corridor apply.c:439
    # end IS_ROCK

    # we should also be aware of the impassable `BOULDER` objects
    # XXX "A boulder blocks your path" hack.c:890 and hack.c:909
    pass

    # begin dangerous cmaps: pools and moats
    S_pool = 32
    S_lava = 34
    S_water = 41
    # end

    # begin ACCESSIBLE as defined in rm.h:94
    S_ndoor = 12
    S_vodoor = 13  # cannot "move diagonally out of an intact doorway."
    S_hodoor = 14

    S_corr, S_litcorr = 21, 22

    # begin IS_ROOM: furniture and room floors
    S_room, S_darkroom = 19, 20
    S_upstair, S_dnstair = 23, 24
    S_upladder, S_dnladder = 25, 26

    S_fountain = 31
    S_throne = 29
    S_sink = 30
    S_grave = 28
    S_altar = 27

    S_ice = 33
    S_vodbridge = 35
    S_hodbridge = 36
    S_air = 39
    S_cloud = 40
    # end IS_ROOM
    # end ACCESSIBLE

    # XXX also check `floodfillchk_match_accessible` from sp_lev.c:3886

    #  /* end dungeon characters, begin traps */

    # see also `trap_types` in include/trap.h, and `glyph_to_trap`
    #  include/rm.h#L629 . we can also `nle.nethack.glyph_to_cmap` into these.
    S_arrow_trap = 42
    S_dart_trap = 43
    S_falling_rock_trap = 44
    S_squeaky_board = 45
    S_bear_trap = 46
    S_land_mine = 47
    S_rolling_boulder_trap = 48
    S_sleeping_gas_trap = 49
    S_rust_trap = 50
    S_fire_trap = 51
    S_pit = 52
    S_spiked_pit = 53
    S_hole = 54
    S_trap_door = 55
    S_teleportation_trap = 56
    S_level_teleporter = 57
    S_magic_portal = 58
    S_web = 59
    S_statue_trap = 60
    S_magic_trap = 61
    S_anti_magic_trap = 62
    S_polymorph_trap = 63
    S_vibrating_square = 64  # /* for display rather than any trap effect */

    #  /* end traps, begin special effects */

    # XXX maybe we should avoid fx like poison gas, booms, or in-flight zaps
    S_vbeam = 65  #  /* The 4 zap beam symbols.  Do NOT separate. */
    S_hbeam = 66  #  /* To change order or add, see function      */
    S_lslant = 67  #  /* zapdir_to_glyph() in display.c.           */
    S_rslant = 68
    S_digbeam = 69  #  /* dig beam symbol */
    S_flashbeam = 70  #  /* camera flash symbol */
    S_boomleft = 71  #  /* thrown boomerang, open left, e.g ')'    */
    S_boomright = 72  #  /* thrown boomerang, open right, e.g. '('  */
    S_ss1 = 73  #  /* 4 magic shield ("resistance sparkle") glyphs */
    S_ss2 = 74
    S_ss3 = 75
    S_ss4 = 76
    S_poisoncloud = 77
    S_goodpos = 78  #  /* valid position for targeting via getpos() */

    #  /* The 8 swallow symbols.  Do NOT separate.  To change order or add, */
    #  /* see the function swallow_to_glyph() in display.c.                 */
    S_sw_tl = 79  #  /* swallow top left [1]             */
    S_sw_tc = 80  #  /* swallow top center [2]    Order: */
    S_sw_tr = 81  #  /* swallow top right [3]            */
    S_sw_ml = 82  #  /* swallow middle left [4]   1 2 3  */
    S_sw_mr = 83  #  /* swallow middle right [6]  4 5 6  */
    S_sw_bl = 84  #  /* swallow bottom left [7]   7 8 9  */
    S_sw_bc = 85  #  /* swallow bottom center [8]        */
    S_sw_br = 86  #  /* swallow bottom right [9]         */

    S_explode1 = 87  #  /* explosion top left               */
    S_explode2 = 88  #  /* explosion top center             */
    S_explode3 = 89  #  /* explosion top right        Ex.   */
    S_explode4 = 90  #  /* explosion middle left            */
    S_explode5 = 91  #  /* explosion middle center    /-\   */
    S_explode6 = 92  #  /* explosion middle right     |@|   */
    S_explode7 = 93  #  /* explosion bottom left      \-/   */
    S_explode8 = 94  #  /* explosion bottom center          */
    S_explode9 = 95  #  /* explosion bottom right           */
    # XXX from nle.nethack import MAXEXPCHARS  # 9 direction incl. center

    #  /* end effects */

    assert MAXPCHARS == 96  # simple version sanity check

    MAXPCHARS = 96  #  /* maximum number of mapped characters */

    condensed_index = {
        el: j
        for j, group in enumerate(
            [
                # walls and generally impassable or locked dungeon features
                [S_stone],
                [S_tree],
                [
                    S_vwall,
                    S_hwall,
                    S_crwall,
                    S_tuwall,
                    S_tdwall,
                    S_tlwall,
                    S_trwall,
                    S_tlcorn,
                    S_trcorn,
                    S_blcorn,
                    S_brcorn,
                ],
                [S_bars],
                [S_vcdoor, S_hcdoor, S_vcdbridge, S_hcdbridge],
                [S_pool, S_water],
                [S_lava],
                # passages
                [S_ndoor, S_vodoor, S_hodoor, S_vodbridge, S_hodbridge],
                [S_corr, S_litcorr],
                [S_room, S_darkroom],
                # XXX what is the gameplay-effective difference between a corrdor,
                #  room floor, and open door lowered bridge? You cannot move
                #  diagonally from a doorway, but then a doorway is a chokepoint
                #  between two walls, and thus can be distinguished from its
                #  surroundings.
                # furniture
                [S_upstair, S_dnstair, S_upladder, S_dnladder],
                [S_fountain],
                [S_throne],
                [S_sink],
                [S_grave],
                [S_altar],
                [S_ice],
                [S_air],
                [S_cloud],
                # traps
                [S_arrow_trap],
                [S_dart_trap],
                [S_falling_rock_trap],
                [S_squeaky_board],
                [S_bear_trap],
                [S_land_mine],
                [S_rolling_boulder_trap],
                [S_sleeping_gas_trap],
                [S_rust_trap],
                [S_fire_trap],
                [S_pit],
                [S_spiked_pit],
                [S_hole],
                [S_trap_door],
                [S_teleportation_trap],
                [S_level_teleporter],
                [S_magic_portal],
                [S_web],
                [S_statue_trap],
                [S_magic_trap],
                [S_anti_magic_trap],
                [S_polymorph_trap],
                [S_vibrating_square],
                # special fx
                [S_vbeam],
                [S_hbeam],
                [S_lslant],
                [S_rslant],
                [S_digbeam],
                [S_flashbeam],
                [S_boomleft],
                [S_boomright],
                [S_ss1],
                [S_ss2],
                [S_ss3],
                [S_ss4],
                [S_poisoncloud],
                [S_goodpos],
                # swallow
                # skipped, since we remap it to dingeon features
                #  with `symbol_sw_to_cmap`
            ]
        )
        for el in group
    }


# remap swallow geometry to cmap symbols
symbol_sw_to_cmap = {
    # insides' geometry is encoded as lower 3bits
    #  1 2 3      0 1 2       +-+
    #  4 5 6 -->> 3 . 4 -->>  | |
    #  7 8 9      5 6 7       +-+
    # see `symbol`
    symbol.S_sw_tl: symbol.S_tlcorn,
    symbol.S_sw_tc: symbol.S_hwall,
    symbol.S_sw_tr: symbol.S_trcorn,
    symbol.S_sw_ml: symbol.S_vwall,
    symbol.S_sw_mr: symbol.S_vwall,
    symbol.S_sw_bl: symbol.S_blcorn,
    symbol.S_sw_bc: symbol.S_hwall,
    symbol.S_sw_br: symbol.S_brcorn,
}


# copied form ./include/hack.h#L370-380
class explosion:
    """explosion types"""

    DARK = 0
    NOXIOUS = 1
    MUDDY = 2
    WET = 3
    MAGICAL = 4
    FIERY = 5
    FROSTY = 6

    from nle.nethack import EXPL_MAX as MAX

    # from src/tielmap.c#L184-189
    _name = tuple(
        map(
            sys.intern,
            (
                "dark",
                "noxious",
                "muddy",
                "wet",
                "magical",
                "fiery",
                "frosty",
            ),
        )
    )


# zap beam types (spell/projectile)
class zap:
    # taken from ./src/zap.c#L38-46
    MAGIC_MISSILE = 0
    FIRE = 1
    COLD = 2
    SLEEP = 3
    DEATH = 4
    LIGHTNING = 5
    POISON_GAS = 6
    ACID = 7
    # XXX potentially can be up to 9 according to ./src/zap.c#L46

    from nle.nethack import NUM_ZAP as MAX

    # taken from ./src/decl.c#L168-185
    _name = tuple(
        map(
            sys.intern,
            (
                "missile",
                "fire",
                "frost",
                "sleep",
                "death",
                "lightning",
                "poison gas",
                "acid",
            ),
        )
    )


# warning levels
# FIXME what is the purpose of these?
class warning:
    # taken from src/drawing.c#L124-137
    WORRY = 0
    CONCERN = 1
    ANXIETY = 2
    DISQUIET = 3
    ALARM = 4
    DREAD = 5

    from nle.nethack import WARNCOUNT as MAX

    _name = tuple(
        map(
            sys.intern,
            (
                "worry",
                "concern",
                "anxiety",
                "disquiet",
                "alarm",
                "dread",
            ),
        )
    )


# taken from mapglyph.c order matters
class glyph_group:
    # See display.h in NetHack.
    MON = 0
    PET = 1
    INVIS = 2
    DETECT = 3
    BODY = 4  # undead are MON
    RIDDEN = 5
    OBJ = 6
    CMAP = 7
    EXPLODE = 8
    ZAP = 9
    SWALLOW = 10
    WARNING = 11
    STATUE = 12

    MAX = 13

    # we do not consider a statue a monster
    ACTORS = frozenset(
        [
            MON,
            PET,
            INVIS,
            DETECT,
            RIDDEN,
        ]
    )

    OBJECTS = frozenset(
        [
            BODY,
            OBJ,
            STATUE,
        ]
    )

    EFFECTS = frozenset(
        [
            EXPLODE,
            ZAP,
            WARNING,
        ]
    )

    LEVEL = frozenset([CMAP, SWALLOW])

    # monsters, special effects and warnings are potentially mobile
    MOBILE = frozenset(
        [
            *ACTORS,
            *EFFECTS,
        ]
    )


# exported macros
class glyph_is:
    # from nle.nethack import glyph_is_monster as monster  # subsumed
    from nle.nethack import glyph_is_normal_monster as normal_monster
    from nle.nethack import glyph_is_pet as pet
    from nle.nethack import glyph_is_body as body
    from nle.nethack import glyph_is_statue as statue
    from nle.nethack import glyph_is_ridden_monster as ridden_monster
    from nle.nethack import glyph_is_detected_monster as detected_monster
    from nle.nethack import glyph_is_invisible as invisible
    from nle.nethack import glyph_is_normal_object as normal_object

    # from nle.nethack import glyph_is_object as obj  # normalize, subsumed

    # ATTN is_trap is off-by-one since it uses NUMTRAPS, which includes NO_TRAP
    from nle.nethack import glyph_is_trap as trap

    from nle.nethack import glyph_is_cmap as cmap
    from nle.nethack import glyph_is_swallow as swallow
    from nle.nethack import glyph_is_warning as warning


# exported macros
class glyph_to:
    from nle.nethack import glyph_to_mon as monster
    from nle.nethack import glyph_to_obj as obj
    from nle.nethack import glyph_to_trap as trap
    from nle.nethack import glyph_to_cmap as cmap
    from nle.nethack import glyph_to_swallow as swallow
    from nle.nethack import glyph_to_warning as warning


def get_group(cls, offset, *symbols):
    return frozenset([offset + getattr(cls, s) for s in symbols])


# index and group
# XXX (group, entity) is enough to disambiguate glyphs
dt_glyph_id = np.dtype(
    [
        ("value", int),  # (backref) the original value of the glyph
        ("group", int),  # the technical group it belongs to
        ("index", int),  # the index within the group
        ("entity", int),  # the semantically unique id
    ]
)


# build the glyph group-index-entity lookup
def glyph_lookup(condensed=False):
    """Returns a lookup table for glyphs."""
    assert not condensed

    # our unique entity index
    n_entity_index = 0

    # reworked table form `nethack_baseline.torchbeast.utils`
    table = np.zeros(MAX_GLYPH + 1, dtype=dt_glyph_id).view(np.recarray)

    # Monsters -- the base category!
    for glyph in range(GLYPH_MON_OFF, GLYPH_PET_OFF):
        table[glyph] = (glyph, glyph_group.MON, glyph - GLYPH_MON_OFF, n_entity_index)
        n_entity_index += 1

    # Pets
    for glyph in range(GLYPH_PET_OFF, GLYPH_INVIS_OFF):
        monster = glyph - GLYPH_PET_OFF
        table[glyph] = (
            glyph,
            glyph_group.PET,
            monster,
            # appropriate normal monster's entity id
            table[monster + GLYPH_MON_OFF].entity,
        )

    # Invisible monsters (only one monster placeholder)
    for glyph in range(GLYPH_INVIS_OFF, GLYPH_DETECT_OFF):
        table[glyph] = (
            glyph,
            glyph_group.INVIS,
            glyph - GLYPH_INVIS_OFF,
            n_entity_index,
        )
        n_entity_index += 1

    # Detected monsters (what are these?)
    for glyph in range(GLYPH_DETECT_OFF, GLYPH_BODY_OFF):
        monster = glyph - GLYPH_DETECT_OFF
        table[glyph] = (
            glyph,
            glyph_group.DETECT,
            monster,
            # appropriate normal monster's entity id
            table[monster + GLYPH_MON_OFF].entity,
        )

    # Bodies of monsters
    for glyph in range(GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF):
        monster = glyph - GLYPH_BODY_OFF
        table[glyph] = (
            glyph,
            glyph_group.BODY,
            monster,
            # appropriate normal monster's entity id (because entities
            #  correspond to information embeddings, whatever form they take)
            table[monster + GLYPH_MON_OFF].entity,
        )

    # Mounts
    for glyph in range(GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF):
        monster = glyph - GLYPH_RIDDEN_OFF
        table[glyph] = (
            glyph,
            glyph_group.RIDDEN,
            monster,
            # appropriate normal monster's entity id
            table[monster + GLYPH_MON_OFF].entity,
        )

    # Items
    for glyph in range(GLYPH_OBJ_OFF, GLYPH_CMAP_OFF):
        table[glyph] = (
            glyph,
            glyph_group.OBJ,
            glyph - GLYPH_OBJ_OFF,
            n_entity_index,
        )
        n_entity_index += 1

    # Dungeon layout
    if not condensed:
        for glyph in range(GLYPH_CMAP_OFF, GLYPH_EXPLODE_OFF):
            table[glyph] = (
                glyph,
                glyph_group.CMAP,
                glyph - GLYPH_CMAP_OFF,  # maps into `symbol`
                n_entity_index,
            )
            n_entity_index += 1

    else:
        # XXX there is no actionalbe information in distinguishing
        #  `([hv]|cr|t[udlr])wall` or `[tb][lr]corn`, i.e. different
        #  wall geometry, and orientation of doors/drawbridges ought
        #  not to have impact on the AI gameplay. There are also lit/dark
        #  glyph symbols `S_corr`, `S_litcorr`, `S_room`, and `S_darkroom`.
        for glyph in range(GLYPH_CMAP_OFF, GLYPH_EXPLODE_OFF):
            cmap = glyph - GLYPH_CMAP_OFF
            table[glyph] = (
                glyph,
                glyph_group.CMAP,
                cmap,
                n_entity_index + symbol.condensed_index[cmap],
            )

        n_entity_index += len(set(symbol.condensed_index.values()))

    # Explosion special fx.: nature x geometry
    for glyph in range(GLYPH_EXPLODE_OFF, GLYPH_ZAP_OFF):
        # get the pure explosion type from the glyph:
        #   nature: see [explosion._name](include/hack.h#L370-380)
        #   geometry [`S_explode[1-9]`](include/rm.h#L216-224)
        nature, geometry = divmod(glyph - GLYPH_EXPLODE_OFF, MAXEXPCHARS)
        # XXX ignore explosions' geometry

        table[glyph] = (
            glyph,
            glyph_group.EXPLODE,
            nature,  # maps into `explosion`
            n_entity_index + nature,  # we need to embed explsion's `nature`
        )

    assert nature + 1 == explosion.MAX
    n_entity_index += explosion.MAX

    # Zap beam special effects: energy x beam
    for glyph in range(GLYPH_ZAP_OFF, GLYPH_SWALLOW_OFF):
        # as in the case of explosions, we infer the `energy` type from zaps
        #   energy: see [ZAP._name](./src/zap.c#L38-46)
        #   beam: (S_[vh]beam|S_[rl]slant) 4 directions
        energy, beam = divmod(glyph - GLYPH_ZAP_OFF, 4)
        # XXX we ignore beam direction

        table[glyph] = (
            glyph,
            glyph_group.ZAP,
            energy,  # maps into `zap`
            n_entity_index + energy,  # energy type is important, eg dath ray
        )

    assert energy + 1 == zap.MAX
    n_entity_index += zap.MAX

    # Swallow map layout: monster x 8 walls for insides' geometry
    # XXX after all being swallowed by a monster is an expectional situation,
    #  which does not benefit from knowing what monster is currently devouring
    #  the player. A panic glyph maybe?
    for glyph in range(GLYPH_SWALLOW_OFF, GLYPH_WARNING_OFF):
        # get the monster type from the surrounding insides
        #   monster: see all monsters [pm.h](./utils/makedefs.c)
        #   layout: S_sw_[tmb][lcr] (excl. S_sw_cc)
        monster, geometry = divmod(glyph - GLYPH_SWALLOW_OFF, 8)

        # just add the offset ./src/mapglyph.c#L101
        cmap = symbol_sw_to_cmap[symbol.S_sw_tl + geometry]

        # XXX do we care about the monster info?
        table[glyph] = (
            glyph,
            glyph_group.SWALLOW,
            cmap,  # maps into `symbol`
            # assume monster's entity id
            table[monster + GLYPH_MON_OFF].entity,
        )

    # Warnings (six different warning levels)
    for glyph in range(GLYPH_WARNING_OFF, GLYPH_STATUE_OFF):
        table[glyph] = (
            glyph,
            glyph_group.WARNING,
            glyph - GLYPH_WARNING_OFF,  # maps into `warning`
            n_entity_index,
        )
        n_entity_index += 1

    # Monster statues
    for glyph in range(GLYPH_STATUE_OFF, MAX_GLYPH):
        monster = glyph - GLYPH_STATUE_OFF
        table[glyph] = (
            glyph,
            glyph_group.STATUE,
            monster,
            # appropriate normal monster's entity id (since monsters may become
            #  statues when hallucinating)
            table[monster + GLYPH_MON_OFF].entity,
        )

    # sanity check
    assert n_entity_index == MAX_ENTITY

    # MAX_GLYPH is the general padding glyph
    table[MAX_GLYPH] = (
        MAX_GLYPH,
        glyph_group.MAX,
        symbol.S_stone,  # map to stone symbol
        MAX_ENTITY,
    )

    return table


glyphlut = glyph_lookup()


# extended lookup table for glyph properties
dt_glyph_ext = np.dtype(
    [
        ("id", dt_glyph_id),
        ("is_background", bool),
        ("is_floodable", bool),
        ("is_accessible", bool),
        ("is_actor", bool),
        ("is_interesting", bool),
        ("screen", "c"),  # the representation of the symbol on the screen
    ]
)


symbol_is_accessible = get_group(
    symbol,
    0,
    # floor, corridor and z-level passages
    "S_ndoor",
    "S_vodoor",
    "S_hodoor",
    "S_vodbridge",
    "S_hodbridge",
    "S_corr",
    "S_litcorr",
    "S_room",
    "S_darkroom",
    "S_upstair",
    "S_dnstair",
    "S_upladder",
    "S_dnladder",
    # furniture
    "S_fountain",
    "S_throne",
    "S_sink",
    "S_grave",
    "S_altar",
    # terrain
    "S_ice",
    "S_air",
    "S_cloud",
)


symbol_is_floodable = get_group(
    symbol,
    0,
    "S_room",
    "S_darkroom",
    # 'S_ndoor',  # an empty doorway is like a floor
    # {S_upstair, "up stairs", "staircase up"},  # from tilemap.c
    # {S_dnstair, "down stairs", "staircase down"},  ditto
    "S_upstair",
    "S_dnstair",
    # {S_upladder, "up ladder", "ladder up"},
    # {S_dnladder, "down ladder", "ladder down"},
    "S_upladder",
    "S_dnladder",
    "S_altar",
    "S_grave",
    "S_throne",
    "S_sink",
    "S_fountain",
)


symbol_is_background = get_group(
    symbol,
    0,
    # walls and obstacles
    "S_stone",
    "S_tree",
    "S_vwall",
    "S_hwall",
    "S_tlcorn",
    "S_trcorn",
    "S_blcorn",
    "S_brcorn",
    "S_crwall",
    "S_tuwall",
    "S_tdwall",
    "S_tlwall",
    "S_trwall",
    "S_bars",
    "S_vodoor",
    "S_hodoor",
    "S_vodbridge",
    "S_hodbridge",
    # corridors and open doors
    "S_ndoor",
    "S_vcdoor",
    "S_hcdoor",
    "S_vcdbridge",
    "S_hcdbridge",
    "S_corr",
    "S_litcorr",
    "S_room",
    "S_darkroom",
    "S_upstair",
    "S_dnstair",
    "S_upladder",
    "S_dnladder",
    # elemental
    "S_pool",
    "S_lava",
    "S_water",
    "S_ice",
    "S_air",
    "S_cloud",
    # furniture
    "S_fountain",
    "S_throne",
    "S_sink",
    "S_grave",
    "S_altar",
)


glyph_is_mundane = get_group(
    symbol,
    GLYPH_CMAP_OFF,
    # stone, trees, walls and other obstacles
    "S_stone",
    "S_tree",
    "S_bars",
    "S_vwall",
    "S_hwall",
    "S_crwall",
    "S_tuwall",
    "S_tdwall",
    "S_tlwall",
    "S_trwall",
    "S_tlcorn",
    "S_trcorn",
    "S_blcorn",
    "S_brcorn",
    # open doors, doorways, lowered brigdes
    "S_ndoor",
    "S_vodoor",
    "S_hodoor",
    "S_vodbridge",
    "S_hodbridge",
    # room floors and corridors
    "S_corr",
    "S_litcorr",
    "S_room",
    "S_darkroom",
    # terrain including lava and water
    "S_ice",
    "S_air",
    "S_cloud",
    "S_lava",
    "S_water",
    "S_pool",
    # all kinds of traps
    "S_arrow_trap",
    "S_dart_trap",
    "S_falling_rock_trap",
    "S_squeaky_board",
    "S_bear_trap",
    "S_land_mine",
    "S_rolling_boulder_trap",
    "S_sleeping_gas_trap",
    "S_rust_trap",
    "S_fire_trap",
    "S_pit",
    "S_spiked_pit",
    "S_hole",
    "S_trap_door",
    "S_teleportation_trap",
    "S_level_teleporter",
    "S_magic_portal",
    "S_web",
    "S_statue_trap",
    "S_magic_trap",
    "S_anti_magic_trap",
    "S_polymorph_trap",
    "S_vibrating_square",
)


# copied from ./src/drawing.c#L139-243
symbol_to_screen = (
    """ |--------||.-|++##..##<><>_|\\#{}.}..## #}"""
    """^^^^^^^^^^^^^^^^^"^^^^~"""
    """|-\\/*!)(0#@*#?"""
    """/-\\||\\-/"""
    """/-\\| |\\-/"""
)


def extended_lut(basic):
    table = np.zeros(len(basic), dtype=dt_glyph_ext).view(np.recarray)
    for j in range(len(table)):
        id = basic[j]

        screen = b"?"
        if id.group in glyph_group.LEVEL:
            screen = symbol_to_screen[id.index]

        table[j] = (
            id,
            (id.group in glyph_group.LEVEL and id.index in symbol_is_background),
            (id.group in glyph_group.LEVEL and id.index in symbol_is_floodable),
            (id.group in glyph_group.LEVEL and id.index in symbol_is_accessible),
            id.group in glyph_group.ACTORS,
            id.value not in glyph_is_mundane,
            screen,
        )

    return table


ext_glyphlut = extended_lut(glyphlut)


# Bottom Line statistics namedtuple, see `./nle/include/nleobs.h#L16-42`
_, blstats_fields = zip(
    *sorted(
        [
            (NLE_BL_X, "x"),
            (NLE_BL_Y, "y"),
            (NLE_BL_STR25, "str"),  # 'strength'
            (NLE_BL_STR125, "strength_percentage"),
            (NLE_BL_DEX, "dex"),  # 'dexterity'
            (NLE_BL_CON, "con"),  # 'constitution'
            (NLE_BL_INT, "int"),  # 'intelligence'
            (NLE_BL_WIS, "wis"),  # 'wisdom'
            (NLE_BL_CHA, "cha"),  # 'charisma'
            (NLE_BL_SCORE, "score"),
            (NLE_BL_HP, "hitpoints"),
            (NLE_BL_HPMAX, "max_hitpoints"),
            (NLE_BL_DEPTH, "depth"),
            (NLE_BL_GOLD, "gold"),
            (NLE_BL_ENE, "energy"),
            (NLE_BL_ENEMAX, "max_energy"),
            (NLE_BL_AC, "armor_class"),
            (NLE_BL_HD, "monster_level"),
            (NLE_BL_XP, "experience_level"),
            (NLE_BL_EXP, "experience_points"),
            (NLE_BL_TIME, "time"),
            (NLE_BL_HUNGER, "hunger_state"),
            (NLE_BL_CAP, "carrying_capacity"),
            (NLE_BL_DNUM, "dungeon_number"),
            (NLE_BL_DLEVEL, "level_number"),
            (NLE_BL_CONDITION, "condition"),
            (NLE_BL_ALIGN, "alignment"),
        ]
    )
)


BLStats = namedtuple("BLStats", blstats_fields)
BLStats.__doc__ += (
    "\n"
    + r"""
    Current bottom line statistics vector.

    Details
    -------
    The descriptions and meanings have been taken from
        [nleobs.h](include/nleobs.h#L16-42)
"""
)


# miscellaneous flags
Misc = namedtuple("Misc", "in_yn_function,in_getlin,xwaitingforspace")
Misc.__doc__ += (
    "\n"
    + r"""
    Miscellaneous flags

    Details
    -------
    The meanings have been taken from [nleobs.h](win/rl/winrl.cc#L290-292).
"""
)
