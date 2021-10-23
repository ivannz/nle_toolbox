import re
from gym import Wrapper

from collections import namedtuple


rx_menu_is_overlay = re.compile(
    r"""
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
    r"""
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

GUIRawMenu = namedtuple('GUIRawMenu', 'n_pages,n_page,is_overlay,data')

GUIMenu = namedtuple('GUIMenu', 'n_pages_left,items,letters')


def menu_extract(lines):
    """Detect the type of the menu (overlay/fullscreen), the number of
    pages, and extract its raw content.
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
    tty_lines = obs['tty_chars'].view('S80')[:, 0]

    # Assume a menu is on the screen. Detect which one (single,
    # multi), (letters if interactive) and extract its content.
    menu = menu_extract([ll.decode('ascii') for ll in tty_lines])
    if menu is None:
        return None

    # extract menu items
    items, letters = [], {}
    for entry in menu.data:
        m = rx_menu_item.match(entry)
        if m is not None:
            lt, it = m.group('letter', 'item')
            items.append(it)
            if lt is not None:
                letters[lt] = it

    # return the parsed menu
    return GUIMenu(
        # number of additional pages
        menu.n_pages - menu.n_page,
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

    return message.rstrip().decode('ascii')


def has_more_messages(obs):
    # get the top line from tty-chars
    # XXX `Misc(*obs['misc']).xwaitingforspace` reacts to menus as well,
    #  bu we want pure multi-part messages.
    return '--More--' in fetch_message(obs, top=True)


class InteractiveWrapper(Wrapper):
    """The base interaction architecture is essentially a middleman, who passes
    the action to the underlying env, but intercepts the resulting transition
    data. It also is allowed, but not obliged to interact with the env, while
    intercapting the observations.
    """
    def reset(self):
        obs, rew, done, info = self.update(self.env.reset(), 0., False, None)
        return obs

    def step(self, action):
        return self.update(*self.env.step(action))

    def update(self, obs, rew=0., done=False, info=None):
        """Perform the necessary updates and envireonment interactions based
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
    """
    def __init__(self, env, *, top=False):
        super().__init__(env)
        self.top = top

    def update(self, obs, rew=0., done=False, info=None):
        # first we detect and parse menus, since messages cannot
        # appear when they are active
        tx = self.fetch_menus(obs, rew, done, info)
        tx = self.fetch_messages(*tx)
        return tx

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
            obs, rew, done, info = self.env.step(' ')  # send SPACE

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
        Single page menus popup in the middle of the terminal ontop of the
        dungeon map (and are sort of `dirty`, meaning that they have arbitrary
        symbols around them), while multi-page menus take up the entire screen
        after clearing it. Overlaid menu regions appear to be right justified,
        while their contents' text is left-justified. All menus are modal, i.e.
        capture the keyboard input until exited. Some menus are static, i.e.
        just display information, while other are interactive, allowing the
        player to select items with letters or punctuation. However, both kinds
        share two special control keys. The space `\\0x20` (`\\040`, 32,
        `<SPACE>`) advances to the next page, or closes the menu, if the page
        was the last or the only one. The escape `\0x1b` (`\033`, 27, `^[`)
        immediately exits any menu.
        """
        page = menu_parse(obs)
        if page is not None:
            # parse menus and collect all their data unless interactive
            pages = []
            while not page.letters and page.n_pages_left > 0:
                pages.append(page)
                obs, rew, done, info = self.env.step(' ')  # send SPACE
                page = menu_parse(obs)

            # no pages until the current page have been interactive, and either
            #  the current page is an interactive one, or the menu has run out
            #  of pages and this is the last page
            pages.append(page)

            # if the current page is non-interactive, then it mustr be the last
            #  one. Send space to close the menu.
            if not page.letters:
                obs, rew, done, info = self.env.step(' ')  # send SPACE

            # join the pages collected so far
            self.menu = GUIMenu(
                page.n_pages_left,
                tuple([it for page in pages for it in page.items]),
                page.letters,
            )

            # if we've got an interactive page the we are mid-menu, otherwise
            #  the extra space sent above has closed the menu.
            self.in_menu = bool(page.letters)

        else:
            self.menu = None
            self.in_menu = False

        # XXX we'd better listen to special character action when
        #  dealing with interactive menus.
        return obs, rew, done, info
