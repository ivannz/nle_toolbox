from collections import namedtuple

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
)


# Bottom Line statistics namedtuple, see `./nle/include/nleobs.h#L16-42`
_, blstats_fields = zip(*sorted([
    (NLE_BL_X,         'x'),
    (NLE_BL_Y,         'y'),
    (NLE_BL_STR25,     'str'),     # 'strength'
    (NLE_BL_STR125,    'strength_percentage'),
    (NLE_BL_DEX,       'dex'),     # 'dexterity'
    (NLE_BL_CON,       'con'),     # 'constitution'
    (NLE_BL_INT,       'int'),     # 'intelligence'
    (NLE_BL_WIS,       'wis'),     # 'wisdom'
    (NLE_BL_CHA,       'cha'),     # 'charisma'
    (NLE_BL_SCORE,     'score'),
    (NLE_BL_HP,        'hitpoints'),
    (NLE_BL_HPMAX,     'max_hitpoints'),
    (NLE_BL_DEPTH,     'depth'),
    (NLE_BL_GOLD,      'gold'),
    (NLE_BL_ENE,       'energy'),
    (NLE_BL_ENEMAX,    'max_energy'),
    (NLE_BL_AC,        'armor_class'),
    (NLE_BL_HD,        'monster_level'),
    (NLE_BL_XP,        'experience_level'),
    (NLE_BL_EXP,       'experience_points'),
    (NLE_BL_TIME,      'time'),
    (NLE_BL_HUNGER,    'hunger_state'),
    (NLE_BL_CAP,       'carrying_capacity'),
    (NLE_BL_DNUM,      'dungeon_number'),
    (NLE_BL_DLEVEL,    'level_number'),
    (NLE_BL_CONDITION, 'condition'),
]))


BLStats = namedtuple('BLStats', blstats_fields)
BLStats.__doc__ += "\n" + r"""
    Current bottom line statistics vector.

    Details
    -------
    The descriptions and meanings have been taken from
        [nleobs.h](include/nleobs.h#L16-42)
"""


# miscellaneous flags
Misc = namedtuple('Misc', 'in_yn_function,in_getlin,xwaitingforspace')
Misc.__doc__ += "\n" + r"""
    Miscellaneous flags

    Details
    -------
    The meanings have been taken from [nleobs.h](win/rl/winrl.cc#L290-292).
"""


def uint8_to_str(
    as_bytes=False, /, *, tty_chars, chars, message, inv_letters, inv_strs, **remaining
):
    """Preprocess all `uint8` arrays to proper `str`, preserving the leading dims."""
    # `tty_chars` is `... x 24 x 80` fixed width string
    tty_chars = tty_chars.view('S80').squeeze(-1)

    # `message` is `... x 256` zero-terminated string
    message = message.view('S256').squeeze(-1)

    # `chars` is `... x 21 x 79` fixed width string (excl. )
    chars = chars.view('S79').squeeze(-1)

    # `inv_letters` is `... x 55` list of single chars (at most 55 items)
    inv_letters = inv_letters.view('c')

    # `inv_strs` is `... x 55 x 80` list of zero-terminated strings
    #  (at most 80 chars per item and at most 55 items)
    inv_strs = inv_strs.view('S80').squeeze(-1)

    # rebuild the kwargs, casting `bytes` to `str` (UCS4 encoding
    #  gives x4 mem blowup!).
    if as_bytes:
        # XXX a tidier `**locals()` also adds unwanted keys, such
        #  as `remaining` and `as_bytes` :(
        return dict(
            tty_chars=tty_chars,
            message=message,
            chars=chars,
            inv_letters=inv_letters,
            inv_strs=inv_strs,
            **remaining,
        )

    return dict(
        tty_chars=tty_chars.astype(str),
        message=message.astype(str),
        chars=chars.astype(str),
        inv_letters=inv_letters.astype(str),
        inv_strs=inv_strs.astype(str),
        **remaining,
    )


def get_bytes(
    *, tty_chars, chars, message, inv_letters, inv_strs, **remaining
):
    return dict(
        tty_chars=bytes(tty_chars),
        message=bytes(message),
        chars=bytes(chars),
        inv_letters=bytes(inv_letters),
        inv_strs=bytes(inv_strs),
        **remaining,
    )
