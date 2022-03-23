"""GUI abstraction and state collection, regex patterns, source code
references, discussion and documentation are valid as of commit
    82bba59dc45ac89844354a819c21e0808168760a
of the NLE's repo.
"""

import re
import numpy as np

from gym import spaces, Wrapper

from itertools import chain
from collections import namedtuple, deque

from warnings import warn

# a pattern to capture features of a modal in-game screen
rx_modal_signature = re.compile(
    rb"""
    (?P<signature>
        # a rare kind of menu, which is actually a message log
        #  we allow at most one white space before `more` signature
        (?P<has_more>\s?--more--)
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

# (20220210) one of the runs failed when skipping an interactible menu caused
#  a second menu to appear (put in and take out from the backpack in one
#  maneuvre). This was caused by incorrectly parsing the item letters by the rx
#  which caused the Chassis to misdetect the first interactive menu, since
#  on L644 the `.in_menu` flag is determined by non-emptiness of `.letters`.
#  This confused the page collator logic.
# seed = 5114808244567105441, 11072120331074183808
#     'lTb'            # pick up coins, take off leather jacket
#     'ahiU $bdefg '   # put the specified uncursed items into the sack
#     'ahbb '          # try to take out coins
#                      <<-- FAILS, unless we add \$ to letter
rx_menu_item = re.compile(
    rb"""
    ^(
        # An interactable menu item begins with a letter and is
        # followed by some whitespace and either a dash or a plus
        # character for unselected and selected items, respectively.
        (?P<letter>[a-z\$\-])
        \s+[\-\+]\s+
        # XXX certain menus appear to be interactible, yet just list the letter
        #  bindings, e.g. the inventory menu.
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
            # The guard in the Vault always asks for the name with a prompt
            # ending with a dash
            #  [invault()#L447](./nle/src/vault.c#L305-552).
            # however sometimes their query also contains a question mark.
            # This is why we prioritize this rx over the next one
            .*-$
        |
            # y/n, direction, object and monster naming, and other prompts,
            # always contain a question mark, e.g.
            #  [do_oname()](./nle/src/do_name.c#L1200-1280) for objects,
            #  [do_mname()](./nle/src/do_name.c#L1118-1196) for monster,
            #  [doengrave()](./nle/src/engrave.c#L1023) for engraving.
            # However, unlike these do-s,
            #  [docall()](./nle/src/do_name.c#L1467-1514)
            # presents the player with a prompt that ends in a colon.
            [^\#].+?[\?:]  # usually prompts come with options after `?`
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

# regexp for those prompts, that had better accept printable characters only
rx_prompt_printable_only = re.compile(
    rb"""
    what\s+
    (command|.*?look\s+for)
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)
# 20220210 [use_crystal_ball](./nle/src/detect.c#L1094-1235) may ask
#    `What do you look for?`
# which accepts a printable symbol, SPACE or ESC. Before doing so it may
# issue a message which looks like
#    `may look for an object or monster symbol.`
# We handle this non-standard prompt by forcing printable symbols.
# seed = 16441573092876245173, 16704658793745847464
#     'bbhjJjJjj,m'  # go to the crystal orb and pick it up
#     'am'           # try to peer into it
#                    # <<-- FALS with an unknown prompt

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
    'GUIModalWindow',
    'is_message,n_row,data,n_pages,n_page',
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


GUIMenuPage = namedtuple(
    'GUIMenuPage',
    'n_page,n_pages,title,items,letters',
)


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
        # FIXME this assumption fails quite often, though
        elif entry and not title and page.n_page == 1:
            title = entry

    # return the parsed menu
    return GUIMenuPage(
        # the current page and the total number of pages
        page.n_page, page.n_pages,
        # the title of the menu (empty if not the first page)
        title,
        # the line-by-line content of the menu's page
        items,
        # which items can be interacted with
        letters,
    )


GUIMenu = namedtuple(
    'GUIMenu',
    'title,n_page,n_pages,content,letters',
)


def join_menu_pages(pages, *, menu=None):
    """Join the auto-collected menu pages with into the current menu.

    Parameters
    ----------
    pages : list of GUIMenuPage
        A list of menu pages automatically collected by skipping with SPACE.
        This means that the page numbers in them are CONTIGUOUS or RESTART at
        page one.

    menu : GUIMenu or None
        The menu state before the current collection of pages was acquired.
        None if there was no menu.

    Returns
    -------
    menu : GUIMenu or None
        The new menu.

    Details
    -------
    We auto-skip the menu's pages consecutively until we either run out of them
    or land on a page that appears to be interactible. In the former case there
    is no menu capturing the screen, while in the latter the menu awaits user's
    interactions, be it closing, navigating between the pages, or engaging with
    the page's content. On our next call, if the interaction caused the menu to
    close or auto-skip until closed, then we will not be joining anything.

    Unless our last action was SPACE, we aren't guaranteed that the menu screen
    has changed. For example in a multi-page inventory menu `i`, the items
    appear to be interactible, when in fact they are not.

    Otherwise, user's actions caused another streak of skippable pages until
    either an interactible page of a menu close event. We treat this new
    streak of pages as belonging to the same menu, which might rarely result
    in several BACK-TO-BACK menus clumped into one. We tacitly assume that
    menus DO NOT spawn at pages other than the FIRST.
    """
    assert pages

    # Detect the last contiguous page sequence. we assume that auto-skipping
    #  never stays on the same page, i.e. either closes, advances, or spawns
    #  a new menu
    j0 = 0
    for j, pg in enumerate(pages):
        # if the current page is the last one, then update the first position
        if pg.n_page == pg.n_pages:
            j0 = j + 1

    if 0 < j0 < len(pages):
        pages = pages[j0:]
        menu = None

    # Check that the auto-skipped pages are contiguous
    page_span = frozenset([pg.n_page for pg in pages])
    assert len(page_span) == len(pages)
    assert frozenset(range(min(page_span), max(page_span) + 1)) <= page_span

    # setup a representation of the span of menu pages
    new = GUIMenu(
        # get the first non-empty title (page order!)
        # XXX This might not be the real title though, since the first page
        #  that we're parsing here might actually be some intermediate one in
        #  the current menu.
        next(filter(bool, (pg.title for pg in pages)), None),
        # the current page and the total number
        pages[-1].n_page, pages[-1].n_pages,
        # the page content
        {pg.n_page: pg.items for pg in pages},
        # cache the letters from the current page
        pages[-1].letters,
    )

    # no need to join menus if the current one was single-paged
    if menu is None or menu.n_pages == 1:
        return new

    # update the multi-page menu
    return GUIMenu(
        # the title stays with us from the very first page
        menu.title,
        # the number of the page currently being displayed
        new.n_page, menu.n_pages,
        # update the current menu's pages with the skipped content
        menu.content | new.content,  # {**menu.content, **new.content}
        # new letters override the older irrelevant ones
        new.letters,
    )


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
        warn("Top-line message extraction is deprecated.", RuntimeWarning)

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
        self.method_ = 'reset'
        try:
            obs, rew, done, info = self.update(
                self.env.reset(), 0., False, None,
            )
            return obs

        finally:
            self.method_ = None

    def step(self, action):
        self.method_ = 'step'
        try:
            return self.update(*self.env.step(action))

        finally:
            self.method_ = None

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
        # make sure that the wrapped env has all the necessary fields
        #  and their dtype is correct. `message` must be np.uint8, since
        #  we use .view('S') to convert to bytes.
        if not ({'message', 'misc'} <= set(env.observation_space)):
            raise RuntimeError("Chassis requires `message` and `misc`"
                               " fields in the observation dict.")

        msg = env.observation_space['message']
        if msg.dtype != np.uint8:
            raise RuntimeError("NLE's `message` field must have"
                               f" `uint8` dtype. Got `{msg.dtype}`.")

        super().__init__(env)

        self.split = split

        # let the user configure SPACE action id
        self.space = space

    def fetch_misc_flags(self, obs):
        """Set attributes from the `misc` field of the received observation.

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

        # we expect this to fail if the `obs['misc']` spec changes upstream
        (
            self.in_yn_function,
            self.in_getlin,
            self.xwaitingforspace
        ) = map(bool, obs['misc'])

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

        # game over screens in NetHack expect SPACE, but they are terminal,
        #  states which makes it irrelevant whether the action is bound or
        #  not. For other game states, such as menus, and multi-part message
        #  log SPACE, ESC, or ENTER are required to unfreeze the game.
        if self.xwaitingforspace and not done and self.space is None:
            raise RuntimeError(
                "NLE is waiting for SPACE, but this action is NOT BOUND."
                " To see what was going on run `obs['tty_chars'].view('S80')`"
                " in a post-mortem debugger, e.g. `import pdb; pdb.pm()`."
            )

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
                    f"Unrecognized screen `{screen}` while waiting for space."
                )

            # the topline message is a zero-terminated ascii string, so we
            #  can trivially test if it is empty.
            is_topl_msg_nonempty = obs['message'][0] != 0
            # XXX In fact [the docs](./nle/doc/window.doc#L46-48) state that
            #  menu/message are mutually exclusive, and forbid using `putstr`
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
                        f"Non-empty message `{messages}` in a menu `{screen}`."
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

        # We've got an alternative here: either the current page is
        #  interactible in which case `in_menu` is set, or currently nothing
        #  is capturing the input, since the game has run out of menu pages
        #  or messages.
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
                        f"Weird prompt `{message}` while expecting user input."
                    )

        # leave out empty messages
        self.messages = tuple(filter(bool, messages))

        # decide if the menus should be joined
        if pages:
            self.menu = join_menu_pages(
                pages, menu=getattr(self, 'menu', None)
            )

        else:
            # no new pages means no menu on screen
            self.menu = None

        # XXX we'd better listen to special character action when
        #  dealing with interactive menus.
        return obs, rew, done, info


def get_wrapper(env, cls=Chassis):
    """Get the specified underlying wrapper."""
    while isinstance(env, Wrapper):
        if isinstance(env, cls):
            return env

        env = env.env

    raise RuntimeError(
        f"The wrapper `{type(env).__name__}` requires"
        f" a `{cls.__name__}` wrapper in the upstream chain."
    )


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
    from nle.nethack import ACTIONS as _raw_nethack_actions

    # carriage return, line feed or escape
    _esc_crlf = frozenset(map(ord, '\033\015\r\n'))

    # carriage return, line feed, space or escape
    _spc_esc_crlf = frozenset(map(ord, ' ')) | _esc_crlf

    # cardinal directions, esc, cr, and lf
    _directions = frozenset(map(ord, 'ykuh.ljbn'))
    _directions_esc_crlf = _directions | _esc_crlf

    # the letters below are always forbidden for neural actors, because they
    #  either have no effect or are useless, given the data in obs, or outright
    #  dangerous.
    _prohibited = frozenset([
        # XXX gym-ids are subject to change depending on the NLE
        # ascii  # char    gym-id  class                   name
        15,      # \\x0f   59      Command                 OVERVIEW
        18,      # \\x12   68      Command                 REDRAW
        33,      # !       85      Command                 SHELL
        246,     # \\xf6   89      Command                 VERSION

        # win by quitting!
        241,     # \\xf1   65      Command                 QUIT

        # we let the chassis handle mores, ESCs and spaces
        13,      # \\r     19      MiscAction              MORE
        27,      # \\x1b   36      Command                 ESC
        32,      # \\x20   99      TextCharacters          SPACE

        # we get attribs from the BLS (although it might be useful to know
        #  our deity, alignment and mission)
        24,      # \\x18   25      Command                 ATTRIBUTES

        # extended commands are handled as composite actions
        35,      # #       20      Command                 EXTCMD

        # these shortcuts are present in `inv_*` fields of the observation
        # XXX but they may be useful for grouping (as they report correct
        #     letter binding).
        34,      # "       77      Command                 SEEAMULET
        40,      # (       82      Command                 SEETOOLS
        41,      # )       84      Command                 SEEWEAPON
        61,      # =       80      Command                 SEERINGS
        91,      # [       78      Command                 SEEARMOR

        # we know our gold from the BLS
        36,      # $       112     TextCharacters          DOLLAR
        36,      # $       79      Command                 SEEGOLD

        # spells can be enumerated from CAST command `Z`, and neural bots
        #  do not need to rebind spell-key mappings
        43,      # +       81      Command                 SEESPELLS
        43,      # +       97      TextCharacters          PLUS

        38,      # &       92      Command                 WHATDOES
        42,      # *       76      Command                 SEEALL
        47,      # /       93      Command                 WHATIS

        # The SEETRAP command simply shows the type of an adjacent trap and
        #  does not `Find traps`. Detecting them is achieved by searching
        #  around for a while with `[0-9]s` (see `SEARCH`).
        94,      # ^       83      Command                 SEETRAP

        # No need for FARLOOK
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
        92,      # \       49      Command                 KNOWN

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

        # jumping is altogether very confusing
        234,     # \\xea   47      Command                 JUMP

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

        # special actions
        233,     # \\xe9   46      Command                 INVOKE
        239,     # \\xef   56      Command                 OFFER
        240,     # \\xf0   62      Command                 PRAY

    ])

    # The following actions, identified by their ASCII code, are the ones that
    #  we allow to the neural agents in the general non-GUI interaction mode (
    #  except for prompt and menu letter interactions).
    _allowed = frozenset([
        # XXX gym-ids are subject to change depending on the NLE
        # ascii ,  # char    gym-id  class                   name

        # inventory management
        65,      # A       89      Command                 TAKEOFFALL
        44,      # ,       61      Command                 PICKUP
        80,      # P       63      Command                 PUTON
        84,      # T       88      Command                 TAKEOFF
        87,      # W       99      Command                 WEAR
        100,     # d       33      Command                 DROP
        236,     # \\xec   52      Command                 LOOT

        # scrolls are consumables, books aren't, but can become useless
        114,     # r       67      Command                 READ
        101,     # e       35      Command                 EAT
        113,     # q       64      Command                 QUAFF

        4,       # \\x04   48      Command                 KICK
        20,      # \\x14   90      Command                 TELEPORT
        58,      # :       51      Command                 LOOK
        70,      # F       39      Command                 FIGHT
        99,      # c       30      Command                 CLOSE
        111,     # o       57      Command                 OPEN
        245,     # \\xf5   96      Command                 UNTRAP
        115,     # s       75      Command                 SEARCH

        # weapon management dual wield, cycle, select and recharge
        88,      # X       95      Command                 TWOWEAPON
        120,     # x       87      Command                 SWAP
        119,     # w       102     Command                 WIELD
        81,      # Q       66      Command                 QUIVER


        # fire readied ammunition from quiver
        102,     # f       40      Command                 FIRE

        # cast a spell from a book
        90,      # Z       28      Command                 CAST

        # use a wand from the inventory z*
        122,     # z       104     Command                 ZAP

        116,     # t       91      Command                 THROW
        237,     # \\xed   53      Command                 MONSTER
        244,     # \\xf4   94      Command                 TURN

        # adventure mode actions
        229,     # \\xe5   37      Command                 ENHANCE
        97,      # a       24      Command                 APPLY
        228,     # \\xe4   32      Command                 DIP
        242,     # \\xf2   71      Command                 RUB
        212,     # \\xd4   92      Command                 TIP
        247,     # \\xf7   103     Command                 WIPE
        210,     # \\xd2   70      Command                 RIDE
        227,     # \\xe3   29      Command                 CHAT
        230,     # \\xe6   41      Command                 FORCE
        243,     # \\xf3   86      Command                 SIT
        112,     # p       60      Command                 PAY

    ])

    # _raw_nethack_actions > _allowed + _prohibited, since compass directions
    #  are automatically accounted for

    def __init__(self, env):
        super().__init__(env)

        # we need a reference to the underlying chassis wrapper
        self.chassis = get_wrapper(env, Chassis)

        # either way let's keep our own copy of the action-ascii pairing
        #  XXX NLE may have different actions corresponding to the same ascii
        self.ascii_to_action = [
            (j, int(a)) for j, a in enumerate(self.unwrapped.actions)
        ]

        # cache the id of the most essential action -- ESCAPE
        self.escape = next((
            a for a, c in self.ascii_to_action if chr(c) == '\033'
        ), None)
        if self.escape is None:
            warn(
                f"NLE `{self.unwrapped}` does not have a bound ESCAPE action.",
                RuntimeWarning,
            )

        # precompute the common masks
        self._allowed_actions = np.array([
            c in self._prohibited for a, c in self.ascii_to_action
        ], dtype=np.int8)

        # printable text and controls
        self._printable_only = np.array([
            not (
                (32 <= c < 128) or c in self._spc_esc_crlf
            ) for a, c in self.ascii_to_action
        ], dtype=np.int8)

        # directions and escapes
        self._directions_only = np.array([
            c not in self._directions_esc_crlf
            for a, c in self.ascii_to_action
        ], dtype=np.int8)

        # properly augment the observation space (assuming the wrapped env is
        #  the NLE). the action space is unchanged.
        if 'action_mask' in self.observation_space.keys():
            raise RuntimeError(
                f"`action_mask` is already declared by `{self.env}`."
            )

        space = spaces.MultiBinary(len(self.ascii_to_action))
        self.observation_space['action_mask'] = space
        # XXX `MultiBinary` has `int8` dtype, which is not exactly `bool`.

        # cache the direction and self action ids
        self.directions = {
            chr(c): a for a, c in self.ascii_to_action if c in self._directions
        }

    def update(self, obs, rew=0., done=False, info=None):
        # after all the possible menu/message interactions have been complete
        #  compute the mask of allowed actions and inject into the `obs`.
        # XXX the mask never forbids the ESC action

        cha = self.chassis
        if cha.in_menu:  # or cha.xwaitingforspace
            # allow escape, space, and the letters
            letters = frozenset(
                chain(self._spc_esc_crlf, map(ord, cha.menu.letters))
            )

            mask = np.array([
                c not in letters for a, c in self.ascii_to_action
            ], dtype=np.int8)

        elif cha.in_getlin:
            # free-form prompt allow all well-behaving chars, ESC and CR
            mask = self._printable_only.copy()
            # XXX certain prompts, like `what command`, do not set `in_getlin`
            #  flag, and instead raise `in_yn_function` for some reason.

        elif cha.in_yn_function:
            match = rx_prompt_options.search(cha.prompt['tail'])
            if match is not None:  # prompt with [...] options
                # [getobj()](./nle/src/invent.c#L1416-1829) prompts always
                #  have a list of options in the tail
                # fetch letters and produce a mask
                letters = decompactify(match.group('options'), b'\033')
                mask = np.array([
                    c not in letters for a, c in self.ascii_to_action
                ], dtype=np.int8)

            elif rx_prompt_what_direction.search(cha.prompt['full']):
                mask = self._directions_only.copy()

            elif rx_prompt_printable_only.search(cha.prompt['full']):
                mask = self._printable_only.copy()

            else:
                # XXX should not be reached by non-prohibited actions, e.g.
                #  the prohibited WHATDOES command '&' yields 'What command?'
                #  prompt, which ends up here.
                raise RuntimeError(f"Unexpected prompt `{cha.prompt['full']}`")

        else:
            # we're in proper gameplay mode, only allow only certain actions
            mask = self._allowed_actions.copy()

        # a numpy mask indicating actions, that SHOULD NOT be taken
        # i.e. masked or forbidden. (0 -- allowed, 1 -- forbidden)
        obs['action_mask'] = mask
        return obs, rew, done, info


class RecentMessageLog(InteractiveWrapper):
    """A non-interactive wrapper that adds a message log to the observations.
    """

    def __new__(cls, env, *, n_recent=0):
        # bypass self if no history is required
        if n_recent < 1:
            return env

        space = env.observation_space['message']
        assert space.dtype == np.uint8

        return object.__new__(cls)

    def __init__(self, env, *, n_recent=0):
        super().__init__(env)

        # we need a reference to the underlying chassis wrapper, since it
        #  colects the messages
        self.chassis = get_wrapper(env, Chassis)
        self.messages = deque([], n_recent)

        # declare the message log observation
        shape = env.observation_space['message'].shape
        self.space = self.observation_space['message_log'] = spaces.Box(
            low=np.iinfo(np.uint8).min, high=np.iinfo(np.uint8).max,
            shape=(n_recent,) + shape, dtype=np.uint8,
        )

    def update(self, obs, rew=0., fin=False, nfo=None):
        # flush the message log
        if self.method_ == 'reset':
            self.messages.extend((b'',) * self.messages.maxlen)

        # get the messages from the Chassis, making sure to include
        #  the empty ones, when there were no messages
        self.messages.extend(self.chassis.messages or (b'',))

        # form the log, convert it to uint8, and add to dict
        log = np.array(self.messages, np.dtype('S256'))
        obs['message_log'] = log[:, np.newaxis].view(np.uint8)

        return obs, rew, fin, nfo
