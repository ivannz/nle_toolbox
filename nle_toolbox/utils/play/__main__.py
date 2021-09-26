import os
import pickle
import sys

import gym
import nle

from time import sleep

from signal import signal, getsignal, SIGINT

from ...wrappers.replay import Replay
from ...bot.genfun import yield_from_nested


def input(prompt=None, *, _input=__builtins__.input):
    """Promt the user with the specified text and default color."""
    if prompt is None:
        return _input()
    return _input('\033[39m\033[29;0H\033[2K\033[m' + str(prompt))


class AutoNLEControls:
    """backward/forward: `, .` by one action, `< >` by ten actions.
    ctrl-c -- stop/resume playback
    `\\000` -- toggle between the human full control and autp playback.
    `q` -- quit
    enter on an empty prompt -- repeat the last input
    """
    playback, handler, pos = True, None, 0

    def __init__(self, env, trace=()):
        self.trace, self.env = trace, env
        self.ctoa = {a: j for j, a in enumerate(env.unwrapped._actions)}

    def step(self, act):
        # control the position in the replayed actions
        if act in ' ,.<>':
            if act in ',<':
                delta = -1 if act == ',' else -10

            elif act in '.>':
                delta = +1 if act == '.' else +10

            else:
                delta = 0

            self.pos = min(max(self.pos + delta, 0), len(self.trace))
            for _ in self.env.replay(
                self.trace[:self.pos],
                seed=self.env._seed,
            ):
                pass

            return True

        # display a basic help at the 37th line
        if act == '?':
            sys.stdout.write('\x1b[37m\x1b[31;0H' + self.__doc__)
            return True

        # unrecognized actions abort the playback altogether
        return False

    def prompt(self, extra=''):
        # prompt for user input or a ctrl+c
        try:
            return input(f"(? for help{extra}) > ")

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

            for c in bytes(ui, 'utf8').decode('unicode-escape'):
                # toggle game/playback control on zero
                if ord(c) == 0:
                    self.playback = not self.playback
                    continue

                # internal playback control
                if self.playback:
                    # abort on invalid command
                    if not self.step(c):
                        return

                    # yield a `None` action to the caler to update the tty
                    obs = yield None

                # external control of the nle (in tha caller)
                else:
                    # the user might have potentially spolied the state
                    self._dirty = True
                    try:
                        # yield the action from the user input and them gobble
                        #  the potetnial more messages
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
                    obs = obs_
                    sleep(delay)

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
