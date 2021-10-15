import numpy as np

from collections import namedtuple

# Bottom Line statistics namedtuple, see `./nle/include/nleobs.h#L16-42`
BLStats = namedtuple('BLStats', [
    'x',
    'y',
    'str',  # 'strength',
    'strength_percentage',
    'dex',  # 'dexterity',
    'con',  # 'constitution',
    'int',  # 'intelligence',
    'wis',  # 'wisdom',
    'cha',  # 'charisma',
    'score',
    'hitpoints',
    'max_hitpoints',
    'depth',
    'gold',
    'energy',
    'max_energy',
    'armor_class',
    'monster_level',
    'experience_level',
    'experience_points',
    'time',
    'hunger_state',
    'carrying_capacity',
    'dungeon_number',
    'level_number',
    'condition',
])
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


def fold2d(
    array,
    /,
    k=1,
    *,
    leading=1,
    writeable=True,
):
    """Zero-copy sliding window view."""
    leading = (leading + array.ndim) if leading < 0 else leading

    if array.ndim < leading + 2:
        raise ValueError(f"No enough dimensions for 2d folding `{array.shape}`.")

    d0, d1, *shape = array.shape[leading:]
    s0, s1, *strides = array.strides[leading:]
    return np.lib.stride_tricks.as_strided(
        array, (
            *array.shape[:leading],
            d0 - 2 * k, d1 - 2 * k,  # n' = n - w + 1, w = k + 1 + k
            k + 1 + k, k + 1 + k,
            *shape,
        ), (
            *array.strides[:leading],
            s0, s1,
            s0, s1,
            *strides,
        ), writeable=writeable,
    )
