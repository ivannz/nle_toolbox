import os
import pickle
import sys
from copy import deepcopy

import pprint as pp
from functools import wraps

import gym
import nle

from time import sleep

from signal import signal, getsignal, SIGINT

from ...wrappers.replay import Replay
from ...bot.genfun import yield_from_nested


def input(prompt=None, *, _input=__builtins__.input):
    """Prompt the user with the specified text and default color."""
    if prompt is None:
        return _input()

    return _input('\033[29;0H\033[2K\r\033[39m\033[m' + str(prompt))


def flush(l=30):
    sys.stdout.write(f'\033[{l};0H\033[2K\r\033[J\033[37m')


class AutoNLEControls:
    """Controls
    `#play` or `\\000` switch playback [A] and human [U] modes
    `enter` repeat last input, `ctrl-c` resume/stops automatic replay
    `ctrl-d` quit, `debug` toggle debug mode, `help` print this help.

    Replay actions backward/forward by one `, .` by ten `< >` in playback mode
    """
    playback, debug, handler, pos = True, False, None, 0

    def __init__(self, env, trace=()):
        self.trace, self.env = trace, env
        self.ctoa = {a: j for j, a in enumerate(env.unwrapped._actions)}

    def step(self, act):
        # control the position in the replayed actions
        if act not in ' ,.<>':
            # unrecognized actions abort the playback altogether
            return None

        if act in ',<':
            delta = -1 if act == ',' else -10

        elif act in '.>':
            delta = +1 if act == '.' else +10

        else:
            delta = 0

        self.pos = min(max(self.pos + delta, 0), len(self.trace))
        for _, _, _, obs, _ in self.env.replay(
            self.trace[:self.pos],
            seed=self.env._seed,
        ):
            pass

        return obs

    def prompt(self, extra=''):
        # prompt for user input or a ctrl+c
        try:
            status = "A" if self.playback else "U"
            status += "D" if self.debug else " "

            return input(f"([{status:2s}] {extra}) > ")

        except KeyboardInterrupt:
            # hook an interrupt handler so that we could stop playback on
            #  the next `ctrl-C`
            self.handler = getsignal(SIGINT)
            signal(SIGINT, self.restore)

    def restore(self, signalnum, frame):
        # restore the original handler whatever it was
        if self.handler is not None:
            signal(SIGINT, self.handler)

        self.handler = None

    def run(self, obs):
        self.pos, self._dirty, ui = 0, False, None
        self.restore(None, None)
        while True:
            # input is None only in the case when the prompt was interrupted
            uip = ui or ''  # previous user input
            ui = self.prompt(f" {{{uip}}}" if uip else "")  # prev ui in braces
            if ui is None:
                # if the user has spoiled the env state by interacting with it
                #  force reset it to the current playback position.
                if self._dirty:
                    self._dirty = False
                    self.step(' ')

                obs = yield self.play(obs)
                continue

            # stick to the previous input if the current is empty
            ui = ui or uip

            # toggle game/playback control on zero byte
            if ui.startswith('#play') or ui == '\000':
                self.playback = not self.playback
                continue  # ignore the rest of the input on mode switch

            if self.playback:
                # special mode toggles
                if 'debug'.startswith(ui):
                    self.debug = not self.debug
                    obs_ = deepcopy(obs)  # make a deep copy, just in case
                    continue

                # display a helpful message
                elif 'help'.startswith(ui):
                    flush(30)
                    print(self.__doc__)
                    continue

                # clear screen
                elif 'clear'.startswith(ui):
                    flush(1)
                    yield None  # just update the tty
                    continue

            # player control mode
            if self.playback:
                # debug mode with `eval`
                if self.debug:
                    flush(30)
                    try:
                        pp.pprint(eval(ui, {}, obs_))  # XXX very dangerous!!!

                    except SyntaxError:
                        # disable debug mode on any syntax error, e.g. escape
                        self.debug = False

                    except Exception as e:
                        print(str(e), type(e))
                        pp.pprint(obs_.keys())

                # internal playback control
                else:
                    for c in bytes(ui, 'utf8').decode('unicode-escape'):
                        # abort on invalid command
                        obs = self.step(c)
                        if obs is None:
                            return

                        # yield a `None` action to the caller to update the tty
                        yield None

            # external control of the nle (in the caller)
            else:
                for c in bytes(ui, 'utf8').decode('unicode-escape'):
                    # the user might have potentially spoiled the state
                    self._dirty = True
                    try:
                        # yield the action from the user input and them gobble
                        #  the potential more messages
                        obs = yield self.ctoa[ord(c)]
                        obs = yield self.skip_mores(obs)

                    except KeyError:
                        if input('Invalid action. abort? [yn] (y)') != 'n':
                            return
                        break

    @property
    def is_auto(self):
        return self.handler is not None

    def play(self, obs):
        while self.is_auto and self.pos < len(self.trace):
            yield self.trace[self.pos]
            self.pos += 1

    def skip_mores(self, obs):
        while b'--More--' in bytes(obs['tty_chars']):
            obs = yield self.ctoa[0o15]  # hit ENTER (can use ESC 0o33)


def replay(filename, delay=0.06, debug=False):
    breakpoint() if debug else None

    state_dict = pickle.load(open(filename, 'rb'))

    sys.stdout.write(
        '\033[2J\033[0;0H'
        f"replaying recording `{os.path.basename(filename)}` "
        f"from `{state_dict['__dttm__']}`\n"
        f"with seeds {state_dict['seed']}."
    )

    # create the env
    env = Replay(gym.make('NetHackChallenge-v0'))

    # force the seed
    env.seed(seed=state_dict['seed'])

    # the player and game controls
    ctrl = AutoNLEControls(env, trace=state_dict['actions'])

    # start the interactive playthrough
    while True:
        # obs mirrors obs_, unless it is the very first iteration!
        obs_, fin = env.reset(), False
        flow, obs = yield_from_nested(ctrl.run(obs_)), None
        try:
            while env.render('human') and not fin:
                act = flow.send(obs)
                if act is not None:
                    obs_, rew, fin, info = env.step(act)
                    sleep(delay)
                obs = obs_

        except EOFError:
            break

        except StopIteration:
            pass

        # an extra prompt to break out from the loop
        if input('restart? [yn] (n)') != 'y':
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactively replay a recorded playthrough.',
        add_help=True)

    parser.add_argument(
        'filename', type=str,
        help='The stored replay data.')

    parser.add_argument(
        '--delay', type=float, default=0.06, required=False, dest='delay',
        help='Delay between steps during replay.')

    parser.add_argument(
        '--debug', required=False, dest='debug', action='store_true',
        help='Enter trace mode.')

    parser.set_defaults(delay=0.06, debug=False)

    args, _ = parser.parse_known_args()
    replay(**vars(args))
