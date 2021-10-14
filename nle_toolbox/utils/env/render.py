import numpy as np

from textwrap import wrap

from ..obs import Misc


def fixup_tty(
    *,
    tty_chars,
    tty_colors,
    tty_cursor,
    message,
    chars,
    colors,
    misc,
    **ignore,
):
    r"""Restore broken `tty_chars` in case of mutliline message with `--More--`.

    Details
    -------
    It apperas that the tty used internally byt the NLE to capture NetHack's
    output operates in CRLF (\r\n) mode, whereas the game itself issues LF (\n).
    This results in broken second and third lines when the message in long
    and was wrapped at 80 char per column by the game. the NLE gives us plenty
    of information to attempt to reconstruct the correct tty-chars output
    """

    # message containing an LF means that originally it did not fit 80 cols
    lf_mask = message == 0x0A
    misc = Misc(*misc.astype(bool))
    if lf_mask.any() and misc.xwaitingforspace:
        # fix the message: replace lf '\n' (\x0a) with whitespace ` ` (\x20)
        message = np.where(lf_mask, 0x20, message)

        # properly wrap the text at 80
        text = message.view('S256')[0].decode('ascii')
        pieces = wrap(text + '--More--', 80)
        if len(pieces) != 2:
            raise RuntimeError(f"Message `{text}` is too long.")

        # recreate tty-chars by patching the relevant regions
        new_tty_chars, new_tty_colors = np.copy(tty_chars), np.copy(tty_colors)

        # fix the game's 21x79 viewport [1:22, :79], multiline messages may
        #  encroach on the top line of the viewport.
        new_tty_chars[1:22, :79] = chars
        new_tty_colors[1:22, :79] = colors

        # patch the top two lines. We may assume the colors are correct, since
        #  the original tty_chars had at least three top lines affected.
        vw_new_tty_chars = new_tty_chars.view('S80')
        for r, line in enumerate(pieces):
            vw_new_tty_chars[r] = bytes(f'{line:<80s}', encoding='ascii')

        # leave the bottom line stats line intact
        pass

        # patch the cursor to the last non-whitespace char in line 1
        new_tty_cursor = np.copy(tty_cursor)
        new_tty_cursor[:] = 1, len(pieces[1])

        # replace with the patched data
        tty_chars, tty_colors = new_tty_chars, new_tty_colors
        tty_cursor = new_tty_cursor

    return dict(
        tty_chars=tty_chars,
        tty_colors=tty_colors,
        tty_cursor=tty_cursor,
    )


def render(
    *,
    tty_chars,
    tty_colors,
    tty_cursor,
    **ignore,
):
    """Render the observation form the NLE using ANSI escapes.

    Details
    -------
    This DOES NOT actually emulate the manner and idiosyncrasies with which
    Nethack actually outputs data to a tty. It employ differential output, i.e.
    it invalidates the affected tty screen region by moving cursor around,
    deleting rows and columns seemingly at will. For example, this although
    this rendered is visually ok, is does not interface well with saiph bot
    for NetHack 3.4:

        https://github.com/canidae/saiph.git

    """
    r, c = tty_cursor

    rows, cols = tty_chars.shape
    ansi = '\033[1;1H\033[2J'
    for i in range(rows):
        for j in range(cols):
            cl, ch = tty_colors[i, j], tty_chars[i, j]

            # use escapes only if necessary
            if not cl:
                ansi += chr(ch)
            else:
                # use separate SGR CSI escapes: attr and color
                ansi += f'\033[{bool(cl&8):d}m\033[3{cl&7:d}m{ch:c}'

        ansi += '\n'

    ansi += f'\033[m\033[{1+r};{1+c}H'

    return ansi
