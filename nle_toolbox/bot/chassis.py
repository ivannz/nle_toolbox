import re
import numpy as np
from gym import Wrapper

from itertools import chain
from collections import namedtuple


rx_menu_is_overlay = re.compile(
    rb"""
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
    \)\s*$
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
    ^(?P<prompt>
        (
            # messages beginning with a hash are considered prompts,
            #  since the game expects input of an extended command
            \#
        |
            # y/n, direction, naming, and other prompts, always contain
            #  a question mark. We look for the first one.
            [^\#][^\?]+\?
        )
    )
    \s*
    (?P<tail>.*)?
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)

# NetHack asks for directions mostly through
#  [getdir(<prompt>)](./nle/src/cmd.c#L5069-5118), however ins special cases
#  it calls its wrapper. For example, [looting](./nle/src/pickup.c#L1888) and
#  [applying](./nle/src/apply.c#L633) use
#  [get_adjacent_loc](./nle/src/cmd.c#L5035-5067), which relies on `getdir`.
# In summery, directional prompts are questions with `what direction`
rx_prompt_what_direction = re.compile(
    rb"""
    what\s+
    direction
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)

# the object selection ui is [getobj()](./nle/src/invent.c#L1416-1829). On
# line [L1654](./nle/src/invent.c#L1654) it forms the query itself from the
# `word` sz with the verb (eat, read, write, wield,, sacrifice etc.) and on
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
rx_prompt_getobj_options = re.compile(
    rb"""
    ^\[
        (
            (?P<options>[a-z\-\#]+)  # available letters L1669
            \s+or\s+
        )?
        [\?\*ynqa]+                  # select from menu L1667
    \]
    """,
    re.VERBOSE | re.IGNORECASE | re.ASCII,
)


GUIRawMenu = namedtuple('GUIRawMenu', 'n_pages,n_page,is_overlay,data')

GUIMenu = namedtuple('GUIMenu', 'n_pages_left,title,items,letters')


def menu_extract(lines):
    """Detect the type of the menu (overlay or full screen), the number
    of pages, and extract its raw content.
    """
    col, row, match = 80, 0, None

    # detect menu box
    matches = map(rx_menu_is_overlay.search, lines)
    for rr, m in enumerate(matches):
        if m is None:
            continue

        beg, end = m.span()
        if beg <= col:
            col, row, match = beg, rr, m

    # extract the menu and the pagination
    if match is None:
        return None

    is_overlay = False
    content = tuple([ll[col:].rstrip() for ll in lines[:row]])
    n_page, n_pages = match.group('cur', 'tot')
    if n_pages is None:
        n_page, n_pages, is_overlay = 1, 1, True

    return GUIRawMenu(
        int(n_pages),
        int(n_page),
        is_overlay,
        content,
    )


def menu_parse(obs):
    """Extract raw data from a menu and enumerate all items, that can
    be interacted with.
    """
    # Assume a menu is on the screen. Detect which one (single,
    # multi), (letters if interactive) and extract its content.
    menu = menu_extract(obs['tty_chars'].view('S80')[:, 0])
    if menu is None:
        return None
    # XXX to make parsing more robust, `.decode('ascii')` was dropped,
    #  see `fetch_message()`

    # extract menu items
    title, items, letters = b'', [], {}
    for entry in menu.data:
        m = rx_menu_item.match(entry)
        if m is not None:
            lt, it = m.group('letter', 'item')
            items.append(it)
            if lt is not None:
                letters[lt] = it

        # the title of the menu is the first non-empty row of its first page
        elif entry and not title and menu.n_page == 1:
            title = entry

    # return the parsed menu
    return GUIMenu(
        # number of additional pages
        menu.n_pages - menu.n_page,
        # the title of the menu (empty if not the first page)
        title,
        # the line-by-line content of the menu
        items,
        # which items can be interacted with
        letters,
    )


def fetch_message(obs, *, top=False):
    if top:
        # padded with whitespace on the right
        message = bytes(obs['tty_chars'][:2])
    else:
        # has trailing zero bytes
        message = bytes(obs['message'].view('S256')[0])

    # XXX we might potentially want to split the message by `\x20\x20`,
    #  because [`update_topl`](./nle/src/topl.c#L255-265) separates multiple
    #  messages, that fin in one line with `  `.

    # It would be nice to use `.decode('ascii')` to aviod dealing with bytes
    #  objects, especially un downtream message consumers and parsers. However
    #  in the case of random exploration, which is a commun use case, decoding
    #  could fail with a `UnicodeDecodeError`, whever the the game ended up
    #  in a user text prompt.
    return message.rstrip()  # .decode('ascii')


def has_more_messages(obs):
    # get the top line from tty-chars
    # XXX `Misc(*obs['misc']).xwaitingforspace` reacts to menus as well,
    #  but we want pure multi-part messages.
    return b'--More--' in fetch_message(obs, top=True)


class InteractiveWrapper(Wrapper):
    """The base interaction architecture is essentially a middleman, who passes
    the action to the underlying env, but intercepts the resulting transition
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
    """Handle multi-part messages, yes-no-s, and other gui events, which
    were not deliberately requested by downstream policies.

    NetHack's gui is not as intricate as in other related games. We need to
    deal with menus, text prompts, messages and y/n questions.

    Attributes
    ----------
    messages : tuple of bytes
        This may be empty if no message was encountered, otherwise it contains
        all pieces of multipart message that the game threw at us due to
        the last action.

    menu : dict
        Either an empty dictionary, which means that the last action did not
        summon a menu, of a non-empty dictionary, with all the menu data
        collected until the menu's last page or a page with interactible items.
        'n_pages_left' indicated the number of pages left in the menu, 'items'
        is the list of all menu items, and 'letters' is a letter-keyed dict of
        items which can be interacted with.

    in_menu : bool
        A handy boolean flag that indicates if the game's gui is currently mid
        menu with an interactible page. Typically the flag is False, because
        it is set when 'letters' in .menu is non-empty.

    prompt : dict
        The current gui prompt. No prompt is empty, otherwise the key 'prompt'
        contains the query, and 'tail' -- the available options or the current
        reply.
    """
    def __init__(self, env, *, top=False, space=' '):
        super().__init__(env)
        self.top = top

        # let the user decide what is the space action's encoding
        self.space = space

    def update(self, obs, rew=0., done=False, info=None):
        # first we detect and parse menus, since messages cannot
        # appear when they are active
        tx = self.fetch_menus(obs, rew, done, info)
        tx = self.fetch_messages(*tx)

        # passive checks and updates
        self.update_prompt(*tx)
        return tx

    def update_prompt(self, obs, rew=0., done=False, info=None):
        """Detect whether NetHack expects some input form a user, either text
        or a response to a yes/no question.
        """
        self.prompt = {}

        # We should detect prompts that do not expect a user input, but look
        #  like ones according to `rx_is_prompt`. A smart way is to figure out
        # if the game's gui is in a [getlin()](./nle/src/ .c#L ) or a `yn`-like
        # function. obs[`misc`] greatly helps us here!
        self.in_yn_function, self.in_getlin, \
            self.xwaitingforspace = map(bool, obs['misc'])

        # nothing to check if we've got no messages, however some messages are
        #  fake prompts, so also try to avoid such cases by inspecting `misc`.
        if self.messages and (self.in_yn_function or self.in_getlin):
            # check for the prompt in the last message
            *ignore, message = self.messages
            match = rx_is_prompt.search(message)
            if match is not None:
                self.prompt = match.groupdict('')

    def fetch_messages(self, obs, rew=0., done=False, info=None):
        """Deal with top line messages

        Details
        -------
        The game reports events, displays status or information in the top two
        lines of the screen. The NLE also provides raw data in the `message`
        field of the observation. When NetHack generally announces in the top
        line, however, if it wants to communicate a single message longer than
        `80` characters, the game allows it to spill over to the second line,
        appending a `--More--` suffix to it. The game does the same if it has
        several short messages to announce. In both cases NetHack's gui expects
        the user to confirm or dismiss each message by pressing Space, Enter
        or Escape.
        """
        buffer = []
        while has_more_messages(obs) and not done:
            # inside this loop the message CANNOT be empty by design
            buffer.append(fetch_message(obs, top=self.top))
            obs, rew, done, info = self.env.step(self.space)  # send SPACE

        # the final message may be empty so we сheck for it
        message = fetch_message(obs, top=self.top)
        if message:
            buffer.append(message)
        self.messages = tuple(buffer)

        # XXX obs['message'] contains the last message
        return obs, rew, done, info

    def fetch_menus(self, obs, rew=0., done=False, info=None):
        """Handle single and multi-page interactive and static menus.

        Details
        -------
        There are two types of menus on NetHack: single paged and multipage.
        Single page menus popup in the middle of the terminal on top of the
        dungeon map (and are sort of `dirty`, meaning that they have arbitrary
        symbols around them), while multi-page menus take up the entire screen
        after clearing it. Overlaid menu regions appear to be right justified,
        while their contents' text is left-justified. All menus are modal, i.e.
        capture the keyboard input until exited. Some menus are static, i.e.
        just display information, while other are interactive, allowing the
        player to select items with letters or punctuation. However, both kinds
        share two special control keys. The space `\\0x20` (`\\040`, 32,
        `<SPACE>`) advances to the next page, or closes the menu, if the page
        was the last or the only one. The escape `\\0x1b` (`\\033`, 27, `^[`)
        immediately exits any menu.
        """
        page = menu_parse(obs)
        if page is not None:
            # get the title from the first page. This might not be the real
            #  title though, since the first page that we're parsing here
            #  might actually be some intermediate page in the current menu.
            title = page.title

            # parse menus and collect all their data unless interactive
            pages = []
            while not page.letters and page.n_pages_left > 0:
                pages.append(page)
                obs, rew, done, info = self.env.step(self.space)  # send SPACE
                page = menu_parse(obs)

            # no pages until the current page have been interactive, and either
            #  the current page is an interactive one, or the menu has run out
            #  of pages and this is the last page
            pages.append(page)

            # if the current page is non-interactive, then it must be the last
            #  one. Send space to close the menu.
            if not page.letters:
                obs, rew, done, info = self.env.step(self.space)  # send SPACE

            # join the pages collected so far
            new_menu = dict(
                title=title,
                n_pages_left=page.n_pages_left,
                items=tuple([it for page in pages for it in page.items]),
                letters=page.letters,
            )

            # check if the current set of pages belong to a menu which was
            #  interupted by an interactible page.
            if self.menu and self.menu['n_pages_left'] > 0:
                new_menu = dict(
                    # the title stays with us from the very first page
                    title=self.menu['title'],
                    # update the page counter
                    n_pages_left=new_menu['n_pages_left'],
                    # join the lists of items
                    items=self.menu['items'] + new_menu['items'],
                    # new letters overrride the older irrelevant ones
                    letters=new_menu['letters'],
                )
            self.menu = new_menu

            # if we've got an interactive page the we are mid-menu, otherwise
            #  the extra space sent above has closed the menu.
            self.in_menu = bool(new_menu['letters'])

        else:
            self.menu = {}
            self.in_menu = False

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
        letters.extend(chain(
            lead[:-1],
            range(ord(lead[-1:]), ord(text[:1]))
        ))
        lead, und, text = text.partition(b'-')
    letters.extend(lead)

    return frozenset(letters)


class ActionMasker(InteractiveWrapper):
    # carriage return, line feed, space or escape
    _spc_esc_crlf = frozenset(map(ord, '\033\015\r\n '))

    # cardinal directions, esc, cr, and lf
    _directions = frozenset(map(ord, 'ykuh.ljbn\033\015\r\n'))

    # the letters below are always forbidden, because they either have no
    #  effect or are useless, given the data in obs, or outright dangerous.
    _prohibited = frozenset([
        18,   # \x12  -- 68 REDRAW
        36,   # $     -- 112 DOLLAR
        38,   # &     -- 92 WHATDOES
        42,   # *     -- 76 SEEALL
        47,   # /     -- 93 WHATIS
        59,   # ;     -- 15 OVERVIEW // No need for farlook
        71,   # G     -- 73 RUSH2
        73,   # I     -- 45 INVENTTYPE
        77,   # M     -- 55 MOVEFAR
        79,   # O     -- 58 OPTIONS
        83,   # S     -- 74 SAVE
        86,   # V     -- 43 HISTORY
        92,   # \\    -- 49 KNOWN
        95,   # _     -- 85 TRAVEL  // fast travel works on landmarks, but
              #                        otherwise consumes slightly more actions
        96,   # `     -- 50 KNOWNCLASS
        103,  # g     -- 72 RUSH
        105,  # i     -- 44 INVENTORY
        109,  # m     -- 54 MOVE
        118,  # v     -- 90 VERSIONSHORT
        191,  # \xbf  -- 21 EXTLIST
        193,  # \xc1  -- 23 ANNOTATE
        225,  # \xe1  -- 22 ADJUST
        246,  # \xf6  -- 89 VERSION
        # 241,  # \xf1  -- 65 QUIT  // win by quitting!
    ])

    def __init__(self, env):
        super().__init__(env)

        # we need a reference to the underlying chassis wrapper
        self.chassis = get_wrapper(env, Chassis)

        # either way let's keep our own copy of ascii to action id mapping
        self.ascii_to_action = {
            int(a): j for j, a in enumerate(self.unwrapped._actions)
        }

        # pre-compute common masks
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
        if cha.in_menu:
            # allow escape, space, or the letters if we're in interactible menu
            letters = frozenset(chain(
                self._spc_esc_crlf, map(ord, cha.menu['letters']),
            ))
            mask = np.array([
                c not in letters for c, a in self.ascii_to_action.items()
            ], dtype=bool)

        elif cha.prompt:
            prompt = cha.prompt['prompt']
            match = rx_prompt_getobj_options.match(cha.prompt['tail'])

            if rx_prompt_what_direction.search(prompt):
                mask = self._directions_only.copy()

            elif match is not None:  # prompt with [...] options
                # [getobj()](./nle/src/invent.c#L1416-1829) prompts always
                #  have a list of options in the tail
                # fetch letters and produce a mask
                letters = decompactify(match.group('options') or b'', b'\033')
                mask = np.array([
                    c not in letters for c, a in
                    self.ascii_to_action.items()
                ], dtype=bool)

            else:
                # free-form prompt allow all well-behaving chars, ESC and CR
                mask = self._printable_only.copy()

        else:
            # we're in proper gameplay mode, only allow only certain actions
            mask = self._allowed_actions.copy()

        # a numpy mask indicating actions, that SHOULD NOT be taken
        # i.e. masked or forbidden.
        obs['chassis_mask'] = mask

        return obs, rew, done, info
