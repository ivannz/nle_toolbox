import numpy as np

from textwrap import wrap

from .defs import Misc


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
    r"""Restore broken `tty_chars` in case of multiline message with `--More--`.

    Details
    -------
    It appears that the tty used internally by the NLE to capture NetHack's
    output operates in CRLF (\r\n) mode, whereas the game itself issues LF
    ("\n"). This results in broken second and third lines when the message in
    long and was wrapped at 80 char per column by the game. the NLE gives us
    plenty of information to attempt to reconstruct the correct tty-chars
    output.
    """

    # message containing an LF means that originally it did not fit 80 cols
    lf_mask = message == 0x0A
    has_any_lf = lf_mask.any()

    # detecting a rare event when the message is a part of a chain, and has
    #  --More--, but fully fits into the top line and thus has no lf in
    # the message. For example, executing 'acy' in a seeded nethack with
    #     seed = 12604736832047991440, 12469632217503715839
    # causes the following message on the top line:
    #   """Raising your wand of slow monster high above your head,
    #   you break it in two!"""
    # which fits on the top line, yet is a multi-part message.
    text = message.view("S256")[0]
    if not has_any_lf and len(text) > 72:
        # we immediately go the the second line and slice from
        # the proper row, since lf had no cr.
        topl = bytes(tty_chars.view("S80")[1:3, 0])[len(text) :]
        has_any_lf = b"--More--" in topl

    misc = Misc(*misc.astype(bool))
    if has_any_lf and misc.xwaitingforspace:
        # fix the message: replace lf '\n' (\x0a) with whitespace ` ` (\x20)
        message = np.where(lf_mask, 0x20, message)

        # properly wrap the text at 80
        text = message.view("S256")[0].decode("ascii")
        pieces = wrap(text + "--More--", 80, break_on_hyphens=False)
        # XXX in the case when multiple engravings are appended to each other
        #  at the same location the message may get so long as to span across
        #  up to four top lines. This is why the check for the number of pieces
        #  has been removed.

        # recreate tty-chars by patching the relevant regions
        new_tty_chars, new_tty_colors = np.copy(tty_chars), np.copy(tty_colors)

        # fix the game's 21x79 viewport [1:22, :79], multi-line messages may
        #  encroach on the top lines of the viewport.
        # XXX The game's map area in `tty_chars` may get `dirty` from long
        #  messages, because it is not redrawn, unless the game receives
        #  `\\x12` REDRAW command. The first two lines are redrawn in `topl`,
        # but the garbage split onto the third and fourth lines is not flushed.
        new_tty_chars[1:22, :79] = chars
        new_tty_colors[1:22, :79] = colors

        # reset the last column
        new_tty_chars[:22, 79:] = 32
        new_tty_colors[:22, 79:] = 0

        # patch the top two lines. We may assume the colors are correct, since
        #  the original tty_chars had at least three top lines affected.
        vw_new_tty_chars = new_tty_chars.view("S80")
        for r, line in enumerate(pieces):
            vw_new_tty_chars[r] = bytes(f"{line:<80s}", encoding="ascii")

            # use black for whitespace
            new_tty_colors[r] = np.where(new_tty_chars[r] == 32, 0, 7)

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
        message=message,
    )


def fixup_message(
    *,
    message,
    misc,
    **ignore,
):
    """Replace line feeds (\\x0a) in the message with whitespace (\\x20)."""

    # message containing an LF means that originally it did not fit 80 cols
    lf_mask = message == 0x0A

    misc = Misc(*misc.astype(bool))
    if lf_mask.any() and misc.xwaitingforspace:
        # fix the message: replace lf '\n' (\x0a) with whitespace ` ` (\x20)
        message = np.where(lf_mask, 0x20, message)

    return dict(message=message)


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
    tty_colors = tty_colors.view(np.uint8)

    # position the cursor at (1, 4) with \033[<L>;<C>H
    ansi = "\033[1;1H\033[2J"
    for i in range(rows):
        for j in range(cols):
            cl, ch = tty_colors[i, j], tty_chars[i, j]

            # use escapes only if necessary
            if not cl:
                ansi += chr(ch)
            else:
                # use separate SGR CSI escapes: attr and color
                # XXX effective until a next SGR ESC
                # XXX `semicolons may separate up to 16 attributes in a seq`
                #  https://man7.org/linux/man-pages/man4/console_codes.4.html
                if bool(cl & 0x80):
                    # Use 8-bit foreground colors
                    ansi += f"\033[38;5;{cl&0x7f:d}m{ch:c}"

                else:
                    # set 3-bit foreground color \033[<bold?>;3<3-bit color>m
                    ansi += f"\033[{bool(cl&8):d};3{cl&7:d}m{ch:c}"

        ansi += "\n"

    # reset the color back to normal, place the cursor
    ansi += f"\033[m\033[{1+r};{1+c}H"

    return ansi


def load_minihack_tileset():
    """Get the tileset provided by minihack."""

    import pickle
    from pkg_resources import resource_filename

    try:
        from minihack.tiles.tile import glyph2tile
        from nle.nethack import MAX_GLYPH

        # make sure all glyphs can be mapped to tiles
        n_extra = max(0, MAX_GLYPH - len(glyph2tile) + 1)
        glyph2tile = np.array(glyph2tile + [-1] * n_extra)

        # array of keys into the tileset dict
        res = resource_filename("minihack.tiles", "tiles.pkl")
        return glyph2tile, pickle.load(open(res, "rb"))

    except ImportError:
        return None


default_tileset = load_minihack_tileset()


def render_to_rgb(glyphs, *, tileset=default_tileset):
    """Render glyphs with the specified tileset."""
    if glyphs.ndim < 2:
        raise TypeError("`glyphs` array must be at least two-dimensional.")

    assert tileset is not None
    glyph2tile, tile2image = tileset

    # the renderer is just a sparse lookup table
    void = np.zeros_like(next(iter(tile2image.values())))
    tiles = np.stack(
        [tile2image.get(tile, void) for tile in glyph2tile[glyphs.flat]], axis=0
    )

    # restore geometry of each frame
    ph, pw, col = void.shape
    *head, gh, gw = glyphs.shape
    frames = tiles.reshape(-1, gh, gw, ph, pw, col)

    # pair related spatial dims and flatten them
    frames = frames.transpose(0, 1, 3, 2, 4, 5)
    return frames.reshape(*head, gh * ph, gw * pw, col)
