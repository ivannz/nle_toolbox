from abc import ABCMeta, abstractmethod
from inspect import isgeneratorfunction
from random import choice


def is_suspended(gen):
    if gen.gi_frame is None:
        return False

    # check the index of the last instruction in the generator's frame.
    #  see `inspect.getgeneratorstate`
    return gen.gi_frame.f_lasti != -1


class Loop(metaclass=ABCMeta):
    @abstractmethod
    def loop(self, rx):
        while False:
            yield None

    def __call__(self, rx):
        yield from self.loop(rx)


class ReactorLoop(Loop):
    def loop(self, rx):
        while self.check(rx):
            rx = yield self.react(rx)

    @abstractmethod
    def check(self, rx):
        return False

    @abstractmethod
    def react(self, rx):
        return None


class FilterLoop(ReactorLoop):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def loop(self, rx):
        try:
            gi = None
            while True:
                if self.check(rx):
                    tx = self.react(rx)

                else:
                    if gi is None:
                        gi, rx = self.wrapped(rx), None
                    tx = gi.send(rx)

                rx = yield tx

        except StopIteration as e:
            raise RuntimeError(e.value) from None


class SplitterLoop(Loop):
    ...


class DispatcherLoop(ReactorLoop):
    def __init__(self, *loops):
        self.loops = loops

    def check(self, rx):
        return True

    def react(self, rx):
        # selection is entirely the dispatcher's job
        # XXX can loops vote/bid?
        return None

        # simply pick a random eligible
        eligible = [
            loop for loop in self.loops
            if getattr(loop, 'check', lambda rx: True)(rx)
            # (what if a loop has no .check?)
        ]

        if eligible:
            return choice(eligible)

    def loop(self, rx):
        try:
            active, gf = {}, self.react(rx)
            while gf is not None:
                if gf not in active:
                    active[gf] = gf(rx)

                gi = active[gf]
                rx = yield gi.send(rx if is_suspended(gi) else None)

                gf = self.react(rx)

        except StopIteration as e:
            raise RuntimeError(e.value) from None


class GobbleMores(FilterLoop):
    def check(self, rx):
        return b'--More--' in bytes(rx['tty_chars'])

    def react(self, rx=None):
        return 0o15


class Human(Loop):
    def loop(self, rx=None):
        input = ''
        while True:
            input = __builtins__.input('> ') or input
            for char in bytes(input, 'utf8').decode('unicode-escape'):
                yield ord(char)


class Control(Loop):
    def __init__(self):
        self.ctrl = GobbleMores(Human())

    def loop(self, rx):
        yield from self.ctrl(rx)


import gym
import nle
from nle_toolbox.wrappers.replay import Replay

ctrl = Control()

try:
    with Replay(gym.make('NetHackChallenge-v0')) as env:
        ctoa = {a: j for j, a in enumerate(env.unwrapped._actions)}

        gi, obs, fin = ctrl(env.reset()), None, False
        while env.render('human') and not fin:
            obs, rew, fin, info = env.step(ctoa[gi.send(obs)])

except StopIteration:
    pass
