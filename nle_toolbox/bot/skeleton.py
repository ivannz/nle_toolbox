from collections import deque

from nle.nethack.actions import Command, MiscAction

from .patterns import ObservationFlags
from ..utils.env.defs import BLStats
from ..utils.env.obs import get_bytes


class Skeleton:
    """The skeleton and the backbone of the bot.

    Supports automatic message skipping.
    """

    def __init__(self, brain, *, auto_more=True, auto_yesman=False):
        self.flags = ObservationFlags()

        self.brain, self.queue = brain, deque()
        self.auto_more, self.auto_yesman = auto_more, auto_yesman

    def reset(self, obs):
        self.brain.reset(obs)
        self.queue.clear()
        self.flags(None)

    def step(self, obs):
        self.flags(get_bytes(**obs))

        # 1. automatic GUI-related actions
        # skip partial info messages in the tty (`--More--`)
        if self.auto_more and self.flags.tty_chars__more_:
            # XXX `ENTER` goes through the messages one-by-one, `ESC` skips all
            return Command.ESC  # MiscAction.MORE

        # 1.5 eternal internal `yes-man`: agree to every action we take
        if self.auto_yesman and self.flags.message__ynq_:
            # XXX the logic here is that we assume intent in all our actions.
            #  Hence, if one caused a prompt, then we must make sure that
            #  the intended action is executed.

            # TODO _most_ yn-questions in NLE are positive, meaning that
            #   `y` agrees, while `n` cancels intent.
            return ord('y')

        # 2. open/closed loop control. Prompt only if we are out of
        #  scheduled actions.
        if self.brain.override(obs):
            self.queue.clear()

        if not self.queue:
            # closed loop control: the brain tells us what to do next
            self.queue.extend(self.brain.step(obs))

        # open loop policy: execute pre-scheduled actions
        if self.queue:
            return self.queue.popleft()

        # just wait
        return ord('.')
