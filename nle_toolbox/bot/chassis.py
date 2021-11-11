"""GUI abstraction and state collection, regex patterns, source code
references, discussion and documentation are valid as of commit
    82bba59dc45ac89844354a819c21e0808168760a
of the NLE's repo.
"""

import re
import numpy as np
from gym import Wrapper

from itertools import chain
from collections import namedtuple


# a pattern to capture features of a modal in-game screen
rx_modal_signature = re.compile(
    rb"""
    (?P<signature>
        # a rare kind of menu, which is actually a message log
        (?P<has_more>--more--)
    |
    \(
        (
            # either we get a short single-page overlay menu
            end
        |
            # or a long multi-page menu
            (?P<cur>\d+)
            \s+ of \s+
            (?P<tot>\d+)
        )
    \)
    )\s*$
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE | re.ASCII,
)

rx_menu_item = re.compile(
    rb"""
    ^(
        # An interactable menu item begins with a letter and if
        # followed by some whitespace and either a dash or a plus
        # character for unselected and selected items, respectively.
        (?P<letter>[a-z])
        \s+[\-\+]\s+
    |
        # An non-interactive menu item starts with whitespace
        \s+
    )
    (?P<item>.*)
    $
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)

rx_is_prompt = re.compile(
    rb"""
    ^(?P<full>
        (?P<prompt>
            # messages beginning with a hash are considered prompts,
            #  since the game expects input of an extended command
            \#
        |
            # y/n, direction, object an monster naming, and other prompts,
            # always contain a question mark, e.g.
            #  [do_oname()](./nle/src/do_name.c#L1200-1280) for objects,
            #  [do_mname()](./nle/src/do_name.c#L1118-1196) for monster,
            #  [doengrave()](./nle/src/engrave.c#L1023) for engraving.
            # However, unlike there do-s
            #  [docall()](./nle/src/do_name.c#L1467-1514)
            # presents the player with a prompt that ends in a colon.
            [^\#][^\?:]+[\?:]
        )
        \s*
        (?P<tail>.*)
    ?)
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)

# NetHack asks for directions mostly through
#     [getdir(<prompt>)](./nle/src/cmd.c#L5069-5118), however in special
# cases it calls its wrapper. For example,
#     [looting](./nle/src/pickup.c#L1888)
# and
#     [applying](./nle/src/apply.c#L633)
# use
#     [get_adjacent_loc](./nle/src/cmd.c#L5035-5067),
# which relies on `getdir`. In summery, directional prompts are questions
#  with `what direction`.
rx_prompt_what_direction = re.compile(
    rb"""
    what\s+
    direction
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)

# the object selection ui is [getobj()](./nle/src/invent.c#L1416-1829). On
# line [L1654](./nle/src/invent.c#L1654) it forms the query itself from the
# `word` sz with the verb (eat, read, write, wield, sacrifice etc.) and on
# line L1670 it invokes [`yn_function`](./nle/src/invent.c#L1670). The loop
# on line [L1488](./nle/src/invent.c#1488-1631) forms the list of
# allowed letters.  [HANDS_SYM](./nle/src/invent.c#14) is '-'.
# XXX trying to detect the getobj prompts is a bit tedious, since in many other
# places the game asks similarly worded `what do you want to` prompt, e.g.
#  [do_oname()](./nle/src/do_name.c#L1200-1280) for objects naming,
#  [do_mname()](./nle/src/do_name.c#L1118-1196) for monster calling, and
#  [doengrave()](./nle/src/engrave.c#L1023) for engraving.
pass

# Perhaps the key distinguishing feature of `getobj` prompts are the letter
# options, listed in its tail after the question marl. Notice, that on line
# [L1639](./nle/src/invent.c#1639) `getobj` calls
# [compactify](./nle/src/invent.c#1353-1359) which replaces consecutive
# letters by spans a-z.
rx_prompt_options = re.compile(
    rb"""
    \[
    # lazily consume all characters between a comma and optional
    #  or help-menu. this has been noticed in only one getlin call
    # in [do_play_instrument](./nle/src/music.c#L715)
    ([^,]+,\s+)?

    (?P<options>.*?)  # available letters L1669, hash [0-9], gold, hands symbol

    # menu options are formed in [`getobj`](./nle/src/invent.c#L1666-1670)
    (\s*(?:or\s+)?[\?\*]+)?
    \]
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)


GUIModalWindow = namedtuple(
    'GUIModalWindow', 'is_message,n_row,data,n_pages,n_page'
)


def extract_modal_window(obs):
    """Detect the type of the modal "window" (overlay or full screen),
    the number of pages if any, and extract its raw content.

    Details
    -------
    In rare circumstances, e.g.  `D,\\015E- something\\015:` dropping all
    items on a tile, then writing something on that tile, and finally reading,
    the game screen is captured by a modal message log. It looks identical to
    an overlay one-page menu except it terminates with a `--More--`  signature,
    and not `(end)`. Unlike menus, an overlaid message log may have an unknown
    number of pages left, which means that we need to keep consuming them until
    no modal window with `--More--` is displayed and prepend the collected data
    to the message, since `more` indicates more 'messages'. At the same time,
    menus always either hint or explicitly display pagination info.

    I suspect that there are no paginated message logs, and if a log is too
    large, then it just spawns many modal windows.

    The least number of item we can drop and still summon such a log seems to
    be `two`. Since each item is reported as one message in the top line, this
    means that a log is created whenever there are at least two simultaneous
    message (not a chain of messages due to elapsed game time).

    seeds = 12301533412141513004, 11519511065143048485
    commands: '@Dg\\015de\\015 E- something\\015:' -- a log with two messages
    commands: '@D,\\015E- something\\015:'         -- many messages

    It seems that a message log's signature is never above the second row,
    unlike multi-part message signature, which shows most often on the first
    line, and rarely on the second one.

    This logic is based more on empirical data about the GUI of ASCII NetHack,
    rather than founded on deep code analysis. The routine `display_nhwindow`
    (aliased `windowprocs.win_display_nhwindow`) is responsible for displaying
    modal (blocking) windows and seems to behave differently between ports.

    Although specific to the architecture of the NLE (nethack interfaces with
    `win/rl` window procs which hook into win/tty emulator, and yield control
    back to python space), it appears that menu signatures (pagination and
    parenthesized `end`) and message log signature (--More--) are generated
    in `process_menu_window` on lines
        [L2007-2013](./nle/win/tty/wintty.c#L1844-2213)
    (and created in [tty_end_menu](./nle/win/tty/wintty.c#L2975-3090) in order
    to get the number of columns). The key variable, `cw->morestr`, is
    displayed at the bottom of an nh-window and defaults to
        [defmorestr](./nle/win/tty/wintty.c#L153).
    Procs such as [look_here#L3525](`./nle/src/o_init.c#L3378-3551) and
        [doclassdisco#L639](`./nle/src/o_init.c#L492-659),
    create message logs by calling `display_nhwindow`, which invokes
    either `process_menu_window` or `process_text_window`, the latter asking
    for [dmore](./nle/win/tty/wintty.c#L1698-1717), which most likely has
    `defmorestr` in `cw->morestr`.
    """
    lines = obs['tty_chars'].view('S80')[:, 0]

    # detect the lower left corner of the modal box by looking for a signature
    col, row, match = 80, 0, None
    for rr, m in enumerate(map(rx_modal_signature.search, lines)):
        if m is None:
            continue

        # we're ok with the leftmost signature match
        beg, _ = m.span()
        if beg <= col:
            col, row, match = beg, rr, m

    if match is None:
        # we couldn't find any signature
        return lines, None

    # get the pagination info and the menu contents, leaving out the signature
    n_page, n_pages = match.group('cur', 'tot')
    content = tuple([ll[col:].rstrip() for ll in lines[:row]])
    return lines, GUIModalWindow(
        # let the rx do the detection for the `is_message` flag
        match.group('has_more') is not None,
        # we need the row number to tell a multi-part message form a log
        row,
        # the lines from the modal box, including empty ones
        content,
        # the total number of pages in a multi-page or overlay menus
        int(n_pages or 1),
        # the current page of the menu
        int(n_page or 1),
    )


GUIMenuPage = namedtuple('GUIMenuPage', 'n_pages_left,title,items,letters')


def parse_menu_page(page):
    """Extract raw data from a menu and enumerate all items, that can
    be interacted with.
    """
    # Assume a menu is on the screen. Detect which one (single,
    # multi), (letters if interactive) and extract its content.
    if page is None:
        return None

    # extract menu items
    # XXX to make parsing more robust, `.decode('ascii')` was dropped,
    #  see `fetch_message()`
    title, items, letters = b'', [], {}
    for entry in page.data:
        m = rx_menu_item.match(entry)
        if m is not None:
            lt, it = m.group('letter', 'item')
            items.append(it)
            if lt is not None:
                letters[lt] = it

        # the title of the menu is the first non-empty row of its first page
        elif entry and not title and page.n_page == 1:
            title = entry

    # return the parsed menu
    return GUIMenuPage(
        # number of additional pages
        page.n_pages - page.n_page,
        # the title of the menu (empty if not the first page)
        title,
        # the line-by-line content of the menu's page
        items,
        # which items can be interacted with
        letters,
    )


def join_menu_pages(pages, *, menu=None):
    """Join the menu pages with into the current menu."""
    if not pages:
        return {}

    if menu is None:
        menu = {}

    # get the first non-empty title
    title = next(filter(bool, (p.title for p in pages)), None)
    # XXX This might not be the real title though, since the first page
    #  that we're parsing here might actually be some intermediate one in
    #  the current menu.

    # combine the items from the pages collected so far
    items = tuple([it for page in pages for it in page.items])
    new_menu = dict(
        title=title,
        items=items,
        n_pages_left=pages[-1].n_pages_left,
        letters=pages[-1].letters,
    )

    # check if the current set of pages belongs to a menu which was
    #  interrupted by a prior interactible page
    if menu and menu['n_pages_left'] > 0:
        new_menu = dict(
            # the title stays with us from the very first page
            title=menu['title'],
            # join the tuples of items
            items=menu['items'] + new_menu['items'],
            # update the page counter
            n_pages_left=new_menu['n_pages_left'],
            # new letters override the older irrelevant ones
            letters=new_menu['letters'],
        )

    return new_menu


def fetch_messages(obs, split=False, top=False):
    """Fetch the messages from the observation's message buffer or screen.

    Details
    -------
    It would be nice to use `.decode('ascii')` to avoid dealing with `bytes`
    objects, especially in downstream message consumers and parsers. However
    in the case of random exploration, which is a common use case, decoding
    could fail with a `UnicodeDecodeError`, whenever the the game ends up
    in a user text prompt.

    If instructed, the message is split by `\\x20\\x20`, because this is how
        [`update_topl`](./nle/src/topl.c#L255-265)
    separates multiple messages, that fit in one line.
    """
    if top:
        topl = obs['tty_chars'].view('S80')[:2, 0]
        message = b' '.join(map(bytes.strip, topl))
        if message.endswith(b'--More--'):
            message = message[:-8].rstrip()

    else:
        message = bytes(obs['message'].view('S256')[0])
        message = message.rstrip()

    # message = message.decode('ascii')
    return message.split(b'  ') if split else [message]


class InteractiveWrapper(Wrapper):
    """The base interaction architecture is essentially a middleman, who passes
    the action to the underlying env and intercepts the resulting transition
    data. It also is allowed, but not obliged to interact with the env, while
    intercepting the observations.
    """
    def reset(self):
        obs, rew, done, info = self.update(self.env.reset(), 0., False, None)
        return obs

    def step(self, action):
        return self.update(*self.env.step(action))

    def update(self, obs, rew=0., done=False, info=None):
        """Perform the necessary updates and environment interactions based
        on the data, intercepted from `.env.step` in response to the action
        most recently sent by the downstream user via our `.step` method.
        """

        raise NotImplementedError

        # update must always return the most recent relevant transition data
        return obs, rew, done, info


class Chassis(InteractiveWrapper):
    """Handle multi-part messages, yes-no-s, and other GUI events, which
    were not deliberately requested by downstream policies.

    NetHack's GUI is not as intricate as in other related games. We need to
    deal with menus, text prompts, messages and y/n questions.

    Attributes
    ----------
    messages : tuple of bytes
        This may be empty if no message was encountered, otherwise it contains
        all pieces of multi-part message that the game threw at us due to
        the last action.

    menu : dict
        Either an empty dictionary, which means that the last action did not
        summon a menu, of a non-empty dictionary, with all the menu data
        collected until the menu's last page or a page with interactible items.
        'n_pages_left' indicated the number of pages left in the menu, 'items'
        is the list of all menu items, and 'letters' is a letter-keyed dict of
        items which can be interacted with.

    prompt : dict
        The current GUI prompt. No prompt is empty, otherwise the key 'prompt'
        contains the query, and 'tail' -- the available options or the current
        reply.

    in_yn_function : bool
        The flag reported by the NLE, which indicates whether the game's GUI
        layer is expecting a choice in a yes/no or multiple-option question.

    in_getlin : bool
        An indicator of the game's expecting a free-form text input from the
        user. Collected from the 'misc' fields of the most recent observation,
        or inferred from the most recent message.

    xwaitingforspace : bool
        This flag is set if the game is expecting the user to acknowledge some
        message or interact with a menu. May be included with the yn-flag.

    in_menu : bool
        A boolean flag that indicates if the game's GUI is currently mid menu
        with an interactible page. Typically the flag is False, because it is
        set when 'letters' in `.menu` is non-empty.
    """
    def __init__(self, env, *, split=True, space=' '):
        super().__init__(env)
        self.split = split

        # let the user decide what is the space action's encoding
        self.space = space

    def fetch_misc_flags(self, obs):
        """Set atributes from the `misc` field of the received observation.

        Details
        -------
        The NLE provides 'misc' field in the observation data, which greatly
        helps with detecting input-capturing GUI screens. Whether the game's
        GUI layer is waiting for user input in `getlin()`
            [rl_getlin](./nle/win/rl/winrl.cc#L1024-1032),
        expecting an valid option in a `yn`-like or multiple choice question
            [rl_yn_function](./nle/win/rl/winrl.cc#L1012-1022),
        or expecting a `space` when reporting a chain of messages (multi-part),
        or a some kind of interaction with a menu's page
            [xwaitforspace](./nle/win/tty/getline.c#L218-249)
        then the respective flags are set. For example, `xwaitforspace` is
        called by
            [more](./nle/win/tty/topl.c#L202-241)
        and
            [dmore](./nle/win/tty/wintty.c#L1698-1717)).
        Note that not all user input capturing states are reflected in these
        flags (find `doextcmd` in a comment inside `.update`).
        """

        self.in_yn_function, \
            self.in_getlin, \
            self.xwaitingforspace = map(bool, obs['misc'])

    def update(self, obs, rew=0., done=False, info=None):
        """Detect whether NetHack expects a text input form the user, a pick
        in a yes/no or a multiple choice question, an acknowledgment of
        a multi-part message, or an interaction with a menu's page.

        Details
        -------
        The recent action may have caused a chain of menus and/or messages.
        We parse the game screen and misc flags and interact with the NLE
        until we land in one of the following states: expecting a text input or
        waiting for a letter choice either in a top line yn-question or within
        an interactible page.

        Dealing with top line and modal message logs
        --------------------------------------------
        The game reports events, displays status or information in the top two
        lines of the screen. The NLE also provides raw data in the `message`
        field of the observation. NetHack generally announces in the top line,
        however, if it wants to communicate a single message longer than `80`
        characters, the NLE's win/tty layer allows a spill over to the second
        line. If the game wants to announce several messages that collectively
        do not fit the top line, then its GUI layer appends a `--More--` suffix
        to the message. In both cases NetHack's GUI expects the user to
        acknowledge or dismiss each message by pressing Space, Enter or Escape.

        In rare circumstances the messages are announced on the screen in
        an overlay modal (blocking) window. See `extract_modal_window`. Also,
            [tty_message_menu](./nle/win/tty/wintty.c#L3135-3167)
        actually makes multi-part messages behave like one-item menus.

        Message handling crucially depends on the TTY data and its correctness.
        See `fixup_tty` in `.utils.env.render`.

        Handling single and multi-page interactive and static menus
        -----------------------------------------------------------
        There are two types of menus on NetHack: single paged and multi-page.
        Single page menus pop-up in the middle of the terminal on top of the
        dungeon map (and are sort of `dirty`, meaning that they have arbitrary
        symbols around them), while multi-page menus take up the entire screen
        after clearing it. Overlain menu regions appear to be right justified,
        while their contents' text is left-justified. All menus are modal, i.e.
        capture the keyboard input until dismissed. Some menus are static, i.e.
        just display information, while other are interactive, allowing the
        player to select items with letters or punctuation. However, both kinds
        share two special control keys. The space `\\0x20` (`\\040`, 32,
        `<SPACE>`) advances to the next page, or closes the menu, if the page
        was the last or the only one. The escape `\\0x1b` (`\\033`, 27, `^[`)
        immediately exits any menu.
        """

        self.in_menu = False
        self.fetch_misc_flags(obs)

        # quit immediately, if we've got an interactible page or we've left
        #  the modal window input capturing section in `win/tty`.
        # XXX `xwaitingforspace` takes priority over other flags and reacts
        #  to menus as well as multi-part messages.
        messages, pages = [], []

        # In rare cases, e.g. drinking a potion of enlightenment, the game
        # may spawn a chain of messages interleaved with overlay menus.
        while not (done or self.in_menu) and self.xwaitingforspace:
            # see if we've got a modal window capturing the game screen. It may
            #  either be a top-line message or an overlay log with `--More--`,
            #  a single-page overlay menu with `(end)`, or a full screen
            #  multi-page menu with pagination info at the bottom.
            screen, modal = extract_modal_window(obs)
            if modal is None:
                raise ValueError(
                    f'Unrecognized screen `{screen}` while waiting for space.'
                )

            # the topline message is a zero-terminated ascii string, so we
            #  can trivially test if it is empty.
            is_topl_msg_nonempty = obs['message'][0] != 0
            # XXX In fact [the docs](./nle/doc/window.doc#L46-48) state that
            #  menu/message are mutually exculsive, and forbid using `putstr`
            #  after [start_menu](./nle/doc/window.doc#L306-310).

            # see if we've got a multi-part top-line message, or a message log:
            #  a topline message appears both in `message` and on the screen,
            #  while the log, being a modal window of type NHW_MENU, shows up
            #  only on the screen.
            self.in_menu = False
            if is_topl_msg_nonempty:
                messages.extend(fetch_messages(obs, self.split))
                if not modal.is_message:
                    raise ValueError(
                        f'Non-empty message `{messages}` in a menu `{screen}`.'
                    )

            elif modal.is_message:
                messages.extend(modal.data)

            else:
                # The detected modal window is a menu
                pages.append(parse_menu_page(modal))
                self.in_menu = bool(pages[-1].letters)

            # request the next screen, unless we're in an interactible page
            if not self.in_menu:
                # the current screen displays a multi-part message or a regular
                # non-interactive menu page. If its the menu, then the page is
                # either the last one, and we need to close the menu, or not,
                # in which case we request the next page. If it is a message
                # then we acknowledge it and ask for the next one.
                obs, rew, done, info = self.env.step(self.space)  # send SPACE
                self.fetch_misc_flags(obs)

        # We've got an alternative here: either the current page is interactible
        #  in which case `in_menu` is set, or currently nothing is capturing
        #  the input, since the game has run out of menu pages or messages.
        self.prompt = {}
        if not self.in_menu:
            # at least one message from the log, or a multi-part message spills
            #  over as a single top-line message.
            messages.extend(fetch_messages(obs, self.split))
            # XXX we should use the unsplit message when detecting the prompt

            # `getlin` is not triggered when an extended command is being input
            self.in_getlin = self.in_getlin or messages[-1].startswith(b'#')
            # XXX This due to [doextcmd](./nle/src/cmd.c#L339-367) turning to
            #  `.win_get_ext_cmd`, that in the NLE is ultimately mapped to
            #  [NetHackRL::rl_get_ext_cmd()](./nle/win/rl/winrl.cc#L1034-1040).
            #  Unlike [NetHackRL::rl_getlin](./nle/win/rl/winrl.cc#L1024-1032),
            #  it does not affect `in_getlin` flag.

            # We should avoid messages that do not expect a user input,
            #  but look like prompts according to `rx_is_prompt`.
            if self.in_yn_function or self.in_getlin:
                message = messages.pop()
                match = rx_is_prompt.search(message)
                if match is not None:
                    self.prompt = match.groupdict('')

                else:
                    raise ValueError(
                        f'Weird prompt `{message}` while expecting user input.'
                    )

        # leave out empty messages
        self.messages = tuple(filter(bool, messages))

        self.menu = join_menu_pages(pages, menu=getattr(self, 'menu', None))
        # XXX we'd better listen to special character action when
        #  dealing with interactive menus.
        return obs, rew, done, info


def get_wrapper(env, cls=Chassis):
    """Get the specified underlying wrapper."""
    while isinstance(env, Wrapper):
        if isinstance(env, cls):
            return env

        env = env.env

    raise RuntimeError


def decompactify(text, defaults=b'\033'):
    """Unpacks sorted letter spans ?-? in bytes-objects into a frozenset.

    Sort of inverse to [compactify](./nle/src/invent.c#1353-1359).
    """
    assert isinstance(text, bytes)
    letters = list(defaults)

    lead, und, text = text.partition(b'-')
    while und:
        letters.extend(lead[:-1])
        # we've got r"^-.*$" -- this means bare hands
        if not lead:
            letters.append(ord(und))

        else:
            letters.extend(range(ord(lead[-1:]), ord(text[:1])))

        lead, und, text = text.partition(b'-')

    letters.extend(lead)

    return frozenset(letters)


class ActionMasker(InteractiveWrapper):
    # carriage return, line feed, space or escape
    _spc_esc_crlf = frozenset(map(ord, '\033\015\r\n '))

    # cardinal directions, esc, cr, and lf
    _directions = frozenset(map(ord, 'ykuh.ljbn\033\015\r\n'))

    # the letters below are always forbidden for neural actors, because they
    #  either have no effect or are useless, given the data in obs, or outright
    #  dangerous.
    _prohibited = frozenset([
        # ascii  # char    gym-id  class                   name
        15,      # \\x0f   59      Command                 OVERVIEW
        18,      # \\x12   68      Command                 REDRAW

        # we let the chassis hande mores, ESCs and spaces
        13,      # \\r     19      MiscAction              MORE
        27,      # \\x1b   36      Command                 ESC
        32,      # \\x20   99      TextCharacters          SPACE

        # we get attribs from the BLS (although it might be useful to know
        #  our deity, alignment and mission)
        24,      # \\x18   25      Command                 ATTRIBUTES
        # extended commands are handled as composite actions
        35,      # #       20      Command                 EXTCMD
        # we know our gold from the BLS
        36,      # $       112     TextCharacters          DOLLAR
        38,      # &       92      Command                 WHATDOES
        42,      # *       76      Command                 SEEALL
        43,      # +       97      TextCharacters          PLUS
        47,      # /       93      Command                 WHATIS
        # No need for farlook
        59,      # ;       42      Command                 GLANCE
        64,      # @       26      Command                 AUTOPICKUP
        67,      # C       27      Command                 CALL
        68,      # D       34      Command                 DROPTYPE
        # we use engrave in composite commands only
        69,      # E       37      Command                 ENGRAVE
        71,      # G       73      Command                 RUSH2
        73,      # I       45      Command                 INVENTTYPE
        77,      # M       55      Command                 MOVEFAR
        79,      # O       58      Command                 OPTIONS
        82,      # R       69      Command                 REMOVE
        83,      # S       74      Command                 SAVE
        86,      # V       43      Command                 HISTORY
        92,      # \\\\    49      Command                 KNOWN
        # NetHack travels by landmarks, but consumes two actions `_:`
        95,      # _       85      Command                 TRAVEL
        96,      # `       50      Command                 KNOWNCLASS
        103,     # g       72      Command                 RUSH
        105,     # i       44      Command                 INVENTORY
        109,     # m       54      Command                 MOVE
        118,     # v       90      Command                 VERSIONSHORT
        191,     # \\xbf   21      Command                 EXTLIST
        193,     # \\xc1   23      Command                 ANNOTATE
        195,     # \\xc3   31      Command                 CONDUCT
        225,     # \\xe1   22      Command                 ADJUST
        # win by quitting!
        241,     # \\xf1   65      Command                 QUIT
        246,     # \\xf6   89      Command                 VERSION

        # as with others, the following commands are useful in special prompts
        34,      # "       101     TextCharacters          QUOTE
        39,      # '       100     TextCharacters          APOS
        45,      # -       98      TextCharacters          MINUS

        # numeric inputs are for specific commands only
        48,      # 0       102     TextCharacters          NUM_0
        49,      # 1       103     TextCharacters          NUM_1
        50,      # 2       104     TextCharacters          NUM_2
        51,      # 3       105     TextCharacters          NUM_3
        52,      # 4       106     TextCharacters          NUM_4
        53,      # 5       107     TextCharacters          NUM_5
        54,      # 6       108     TextCharacters          NUM_6
        55,      # 7       109     TextCharacters          NUM_7
        56,      # 8       110     TextCharacters          NUM_8
        57,      # 9       111     TextCharacters          NUM_9
    ])

    def __init__(self, env):
        super().__init__(env)

        # we need a reference to the underlying chassis wrapper
        self.chassis = get_wrapper(env, Chassis)

        # either way let's keep our own copy of the ascii to action id mapping
        self.ascii_to_action = {
            int(a): j for j, a in enumerate(self.unwrapped._actions)
        }

        # precompute common masks
        self._allowed_actions = np.array([
            c in self._prohibited for c, a in self.ascii_to_action.items()
        ], dtype=bool)

        # printable text and controls
        self._printable_only = np.array([
            not (
                (32 <= c < 128) or c in self._spc_esc_crlf
            ) for c, a in self.ascii_to_action.items()
        ], dtype=bool)

        # directions and escapes
        self._directions_only = np.array([
            c not in self._directions
            for c, a in self.ascii_to_action.items()
        ], dtype=bool)

    def update(self, obs, rew=0., done=False, info=None):
        # after all the possible menu/message interactions have been complete
        #  compute the mask of allowed actions and inject into the `obs`.
        # XXX the mask never forbids the ESC action

        cha = self.chassis
        if cha.in_menu:  # or cha.xwaitingforspace
            # allow escape, space, and the letters
            letters = frozenset(
                chain(self._spc_esc_crlf, map(ord, cha.menu['letters']))
            )

            mask = np.array([
                c not in letters for c, a in self.ascii_to_action.items()
            ], dtype=bool)

        elif cha.in_yn_function:
            match = rx_prompt_options.search(cha.prompt['tail'])
            if match is not None:  # prompt with [...] options
                # [getobj()](./nle/src/invent.c#L1416-1829) prompts always
                #  have a list of options in the tail
                # fetch letters and produce a mask
                letters = decompactify(match.group('options'), b'\033')
                mask = np.array([
                    c not in letters for c, a in self.ascii_to_action.items()
                ], dtype=bool)

            elif rx_prompt_what_direction.search(cha.prompt['full']):
                mask = self._directions_only.copy()

            else:
                raise RuntimeError

        elif cha.in_getlin:
            # free-form prompt allow all well-behaving chars, ESC and CR
            mask = self._printable_only.copy()

        else:
            # we're in proper gameplay mode, only allow only certain actions
            mask = self._allowed_actions.copy()

        # a numpy mask indicating actions, that SHOULD NOT be taken
        # i.e. masked or forbidden.
        obs['chassis_mask'] = mask

        return obs, rew, done, info
