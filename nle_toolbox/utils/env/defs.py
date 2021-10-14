"""Various useful defs, macros and logic mined from NetHack and the NLE.
"""
# flake8: noqa
# pycodestyle: noqa
# ignore pep violations since this is mostly copied from c
#   E221 - multiple spaces before operator
#   E114 - indentation is not a multiple of four (comment)
#   E116 - unexpected indentation (comment)


# offsets
from nle.nethack import (
    GLYPH_MON_OFF,      # a monster
    GLYPH_PET_OFF,      # a pet
    GLYPH_INVIS_OFF,    # invisible
    GLYPH_DETECT_OFF,   # mon detect
    GLYPH_BODY_OFF,     # a corpse
    GLYPH_RIDDEN_OFF,   # mon ridden
    GLYPH_OBJ_OFF,      # object
    GLYPH_CMAP_OFF,     # cmap
    GLYPH_EXPLODE_OFF,  # explosion
    GLYPH_ZAP_OFF,      # zap beam
    GLYPH_SWALLOW_OFF,  # swallow
    GLYPH_WARNING_OFF,  # warn flash
    GLYPH_STATUE_OFF,   # a statue

    MAX_GLYPH,          # (end)
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


# the number of distinct entities that glyphs actually correspond to
MAX_ENTITY = (
    NUMMONS                      # normal monsters
    + 0                          # pets: mapped to normal monsters
    + 1                          # single invisible monster placeholder
    + 0                          # detected (?) monsters: mapped to normal
    + 0                          # coprses: mapped to normal (not undead)
    + 0                          # mounts: mapped to monsters
    + NUM_OBJECTS                # in-game items
    + (MAXPCHARS - MAXEXPCHARS)  # level topology, walls, corridors, doors etc.
    + EXPL_MAX                   # explosion types: ignoring geometry
    + NUM_ZAP                    # zap (spells projectiles): ignoring direction
    + 0                          # monsters' insides: mapped to normal monsters
    + WARNCOUNT                  # danger vibes from unknown monsters
    + 0                          # monster statues: mapped to normal monsters
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
    CORPSE   = 0x01
    INVIS    = 0x02
    DETECT   = 0x04
    PET      = 0x08
    RIDDEN   = 0x10
    STATUE   = 0x20

    # boulders hide objpiles, corpses and statues count towards objpiles
    # NB affected by hallucination src/maglyph.c#L57-58
    OBJPILE  = 0x40  # /* more than one stack of objects */

    # determined by the glyph and and whether the color is being used and
    #  the water/lava symbols coincide.
    BW_LAVA  = 0x80  # /* 'black & white lava': highlight lava if it
                     #    can't be distringuished from water by color */


# cmap symbol-glyph semantics
class symbol:
    """Copied verbatim and REORDERED from include/rm.h"""
    # /* begin dungeon characters */

    # begin IS_ROCK according to rm.h:89 and hack.c `test_move` (713), `bad_rock` (659)
    # XXX all are impassable, unless the hero is polymorphed or etheral,
    #  can tunnel or has a pick axe (digging implement). See also mondata.c and
    S_stone     =  0
    S_tree      = 18  # /* KMH */  # passable if hero has an axe, but not ethereal.
                      # XXX Replaces stone in arboreal dungeons. Can also be a standalone tree.
    # begin walls
    S_vwall     =  1
    S_hwall     =  2
    S_tlcorn    =  3
    S_trcorn    =  4
    S_blcorn    =  5
    S_brcorn    =  6
    S_crwall    =  7
    S_tuwall    =  8
    S_tdwall    =  9
    S_tlwall    = 10
    S_trwall    = 11
    # end walls

    # XXX also passable if hero can corrode, ooze or amorphous
    S_bars      = 17  # /* KMH -- iron bars */

    # XXX also passable if hero can ooze or amorphous
    S_vcdoor    = 15  # /* closed door, vertical wall */
    S_hcdoor    = 16  # /* closed door, horizontal wall */

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
    S_pool      = 32
    S_lava      = 34
    S_water     = 41
    # end

    # begin ACCESSIBLE as defined in rm.h:94
    S_ndoor     = 12
    S_vodoor    = 13  # cannot "move diagonally out of an intact doorway."
    S_hodoor    = 14

    S_corr, S_litcorr = 21, 22

    # begin IS_ROOM: furniture and room floors
    S_room, S_darkroom = 19, 20
    S_upstair, S_dnstair = 23, 24
    S_upladder, S_dnladder = 25, 26

    S_fountain  = 31
    S_throne    = 29
    S_sink      = 30
    S_grave     = 28
    S_altar     = 27

    S_ice       = 33
    S_vodbridge = 35
    S_hodbridge = 36
    S_air       = 39
    S_cloud     = 40
    # end IS_ROOM
    # end ACCESSIBLE

    # XXX also check `floodfillchk_match_accessible` from sp_lev.c:3886

    #  /* end dungeon characters, begin traps */

    # see also `trap_types` in include/trap.h, and glyph_to_trap include/rm.h#L629
    # we can `nle.nethack.glyph_to_cmap` into this
    S_arrow_trap           = 42
    S_dart_trap            = 43
    S_falling_rock_trap    = 44
    S_squeaky_board        = 45
    S_bear_trap            = 46
    S_land_mine            = 47
    S_rolling_boulder_trap = 48
    S_sleeping_gas_trap    = 49
    S_rust_trap            = 50
    S_fire_trap            = 51
    S_pit                  = 52
    S_spiked_pit           = 53
    S_hole                 = 54
    S_trap_door            = 55
    S_teleportation_trap   = 56
    S_level_teleporter     = 57
    S_magic_portal         = 58
    S_web                  = 59
    S_statue_trap          = 60
    S_magic_trap           = 61
    S_anti_magic_trap      = 62
    S_polymorph_trap       = 63
    S_vibrating_square     = 64  #  /* for display rather than any trap effect */

    #  /* end traps, begin special effects */

    # XXX maybe we should avoid fx like poison gas, booms, or in-flight zaps
    S_vbeam     = 65  #  /* The 4 zap beam symbols.  Do NOT separate. */
    S_hbeam     = 66  #  /* To change order or add, see function      */
    S_lslant    = 67  #  /* zapdir_to_glyph() in display.c.           */
    S_rslant    = 68
    S_digbeam   = 69  #  /* dig beam symbol */
    S_flashbeam = 70  #  /* camera flash symbol */
    S_boomleft  = 71  #  /* thrown boomerang, open left, e.g ')'    */
    S_boomright = 72  #  /* thrown boomerang, open right, e.g. '('  */
    S_ss1       = 73  #  /* 4 magic shield ("resistance sparkle") glyphs */
    S_ss2       = 74
    S_ss3       = 75
    S_ss4       = 76
    S_poisoncloud = 77
    S_goodpos   = 78  #  /* valid position for targeting via getpos() */

    #  /* The 8 swallow symbols.  Do NOT separate.  To change order or add, */
    #  /* see the function swallow_to_glyph() in display.c.                 */
    S_sw_tl     = 79  #  /* swallow top left [1]             */
    S_sw_tc     = 80  #  /* swallow top center [2]    Order: */
    S_sw_tr     = 81  #  /* swallow top right [3]            */
    S_sw_ml     = 82  #  /* swallow middle left [4]   1 2 3  */
    S_sw_mr     = 83  #  /* swallow middle right [6]  4 5 6  */
    S_sw_bl     = 84  #  /* swallow bottom left [7]   7 8 9  */
    S_sw_bc     = 85  #  /* swallow bottom center [8]        */
    S_sw_br     = 86  #  /* swallow bottom right [9]         */

    S_explode1  = 87  #  /* explosion top left               */
    S_explode2  = 88  #  /* explosion top center             */
    S_explode3  = 89  #  /* explosion top right        Ex.   */
    S_explode4  = 90  #  /* explosion middle left            */
    S_explode5  = 91  #  /* explosion middle center    /-\   */
    S_explode6  = 92  #  /* explosion middle right     |@|   */
    S_explode7  = 93  #  /* explosion bottom left      \-/   */
    S_explode8  = 94  #  /* explosion bottom center          */
    S_explode9  = 95  #  /* explosion bottom right           */
    # XXX from nle.nethack import MAXEXPCHARS  # 9 direction incl. center

    #  /* end effects */

    assert MAXPCHARS == 96  # simple version sanity check

    MAXPCHARS   = 96  #  /* maximum number of mapped characters */


# copied form ./include/hack.h#L370-380
class explosion:
    """explosion types"""
    DARK    = 0
    NOXIOUS = 1
    MUDDY   = 2
    WET     = 3
    MAGICAL = 4
    FIERY   = 5
    FROSTY  = 6

    from nle.nethack import EXPL_MAX as MAX

    # from src/tielmap.c#L184-189
    _name = tuple(map(sys.intern, (
        "dark",
        "noxious",
        "muddy",
        "wet",
        "magical",
        "fiery",
        "frosty",
    )))


# zap beam types (spell/projectile)
class zap:
    # taken from ./src/zap.c#L38-46
    MAGIC_MISSILE = 0
    FIRE          = 1
    COLD          = 2
    SLEEP         = 3
    DEATH         = 4
    LIGHTNING     = 5
    POISON_GAS    = 6
    ACID          = 7
    # XXX potentially can be up to 9 according to ./src/zap.c#L46

    from nle.nethack import NUM_ZAP as MAX

    # taken from ./src/decl.c#L168-185
    _name = tuple(map(sys.intern, (
        "missile",
        "fire",
        "frost",
        "sleep",
        "death",
        "lightning",
        "poison gas",
        "acid",
    )))


# FIXME not sure what this is
class warning:
    # taken from src/drawing.c#L124-137
    WORRY = 0
    CONCERN = 1
    ANXIETY = 2
    DISQUIET = 3
    ALARM = 4
    DREAD = 5

    from nle.nethack import WARNCOUNT as MAX

    _name = tuple(map(sys.intern, (
        "worry",
        "concern",
        "anxiety",
        "disquiet",
        "alarm",
        "dread",
    )))


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
