def render(tty_colors, tty_chars, tty_cursor, **ignore):
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
