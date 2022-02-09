import os
import pickle
import sys
from copy import deepcopy

import pprint as pp

import gym
import nle

from time import sleep

from signal import signal, getsignal, SIGINT

from .wrapper import Replay, ReplayToFile
from ..env.render import fixup_tty, render as render_obs


def input(prompt=None, *, _input=__builtins__.input):
    """Prompt the user with the specified text and default color."""
    if prompt is None:
        return _input()

    return _input('\033[29;0H\033[2K\r\033[39m\033[m' + str(prompt))


def flush(line=30):
    sys.stdout.write(f'\033[{line};0H\033[2K\r\033[J\033[37m')


class AutoNLEControls:
    """Controls
    `play` or `<ESC>`x3 to switch playback [A] and human control [U] modes
    `enter` repeat last input, `ctrl-c` resume/stops automatic replay
    `ctrl-d` quit, `debug` toggle debug mode, `help` print this help

    Replay actions backward/forward by one `, .` by ten `< >` in playback
    mode, `signed integer` moves to position from beginning/end.
    """
    playback, debug, handler, pos = True, False, None, 0

    def __init__(self, env, trace=()):
        self.trace, self.env = trace, env
        self.ctoa = {a: j for j, a in enumerate(env.unwrapped._actions)}
        self.handler = None

    def step(self, act):
        # control the position in the replayed actions
        if isinstance(act, int):
            # integers are used for wraparound indexing
            pos = len(self.trace) + act if act < 0 else act

        elif act in ' ,.<>':
            if act in ',<':
                delta = -1 if act == ',' else -10

            elif act in '.>':
                delta = +1 if act == '.' else +10

            else:
                delta = 0

            pos = self.pos + delta

        # unrecognized actions abort the playback altogether
        else:
            raise ValueError

        self.pos = min(max(pos, 0), len(self.trace))
        for _, _, _, obs, _ in self.env.replay(
            self.trace[:self.pos],
            seed=self.env._seed,
        ):
            pass

        return obs

    def user_input(self):
        ui, history, hix = None, [], 0
        while True:
            # force reset the standby flag
            self.restore()

            # get the user control, while showing the previous user input
            prev_ui = ui or ''
            try:
                status = 'A' if self.playback else 'U'
                status += 'D' if self.debug else ' '
                extra = f' {{{prev_ui}}}' if prev_ui else ''

                # prompt for user input or intercept a ctrl+C
                ui = input(f"([{self.pos:5d} {status:2s}] {extra}) > ")

            except KeyboardInterrupt:
                # hook an interrupt handler so that we could stop playback on
                #  the next `ctrl-C`
                self.handler = getsignal(SIGINT)
                signal(SIGINT, self.restore)

                yield None
                continue

            # replace the user input with the one logged in the history
            #  on up/down keys
            if ui.startswith(('\x1b[A', '\x1b[B')):
                hix = hix + (-1 if ui == '\x1b[A' else +1)
                hix = max(0, min(len(history) - 1, hix))
                ui = history[hix] if history else ''
                continue

            if ui and (not history or history[-1] != ui):
                history.append(ui)

            hix = len(history) - 1

            # stick to the previous input if the current is empty
            ui = ui or prev_ui
            if ui:
                yield ui

    def restore(self, signalnum=None, frame=None):
        # restore the original handler whatever it was
        if self.handler is not None:
            signal(SIGINT, self.handler)

        self.handler = None

    @property
    def standby(self):
        return self.handler is None

    def __iter__(self):
        self.pos, self._dirty = 0, False

        # start in playback standby mode
        self.playback = True
        user_input = self.user_input()

        # reset the env
        obs = self.env.reset()
        yield obs

        while True:
            # auto play until interrupted or exhausted the trace
            while self.playback and not self.standby and (
                self.pos < len(self.trace)
            ):
                obs, rew, fin, nfo = self.env.step(self.trace[self.pos])
                self.pos += 1

                # yield `obs` to the caller to update the tty
                yield obs

            # ensure the ctrl+C handler is reset if we stopped
            self.restore()

            # get the user control
            ui = next(user_input)

            # input is `None` only when the prompt was interrupted by ctrl+C
            if ui is None:
                # if the user has spoiled the env state by interacting with it
                #  force reset it to the current playback position.
                if self._dirty:
                    obs = self.step(' ')
                    self._dirty = False

                self.playback = True
                continue

            # special commands are prefixed with double-escape in control mode
            if self.playback or ui.startswith('\033\033\033'):
                if ui.startswith('\033\033\033'):
                    ui = ui[3:]

                # toggle game/playback control on zero byte
                if not ui or 'play'.startswith(ui):
                    self.playback = not self.playback
                    self.debug = False
                    continue  # ignore the rest of the input on mode switch

                # debug mode toggle
                elif 'debug'.startswith(ui):
                    self.debug = not self.debug
                    continue

                # display a helpful message
                elif 'help'.startswith(ui):
                    flush(30)
                    print(self.__doc__)
                    continue

                # clear screen
                elif 'seed'.startswith(ui):
                    flush(30)
                    print(self.env._seed)
                    continue

                # clear screen
                elif 'clear'.startswith(ui):
                    flush(1)
                    yield obs
                    continue

                # nothing to handle: fall through to the actual logic
                pass

            # debug mode with `eval` in both playback and control modes
            if self.debug:
                flush(30)
                try:
                    # XXX very dangerous!!!
                    pp.pprint(eval(ui, {}, deepcopy(obs)))

                except SyntaxError:
                    # disable debug mode on any syntax error, e.g. escape
                    self.debug = False

                except Exception as e:
                    print(str(e), type(e))

                continue

            # if we're in playback mode then, first, try to interpret the input
            #  as the index in the trace
            if self.playback:
                try:
                    obs = self.step(int(ui))
                    yield obs
                    continue

                except ValueError:
                    pass

            # the user might have potentially spoiled the state
            self._dirty = not self.playback
            for c in bytes(ui, 'utf8').decode('unicode-escape'):
                try:
                    # playback controls
                    if self.playback:
                        obs = self.step(c)

                    # game control
                    else:
                        obs, rew, fin, nfo = self.env.step(self.ctoa[ord(c)])

                    yield obs

                except (KeyError, ValueError):
                    # abort on invalid command
                    if input('Invalid command. abort? [yn] (y)') != 'n':
                        return

                    break


def render(obs):
    if obs is not None:
        sys.stdout.write(render_obs(**fixup_tty(**obs)))
        sys.stdout.flush()
        return obs


def replay(filename, delay=0.06, debug=False, seed=None):
    breakpoint() if debug else None

    state_dict = pickle.load(open(filename, 'rb'))

    sys.stdout.write(
        "\033[2J\033[0;0H replaying recording `{os.path.basename(filename)}`"
        f" from `{state_dict['__dttm__']}`\nwith seeds {state_dict['seed']}."
    )

    # create the env and force the seed
    env = Replay(gym.make('NetHackChallenge-v0'))
    env.seed(seed=state_dict['seed'])

    # the player and game controls
    ctrl = AutoNLEControls(env, trace=state_dict['actions'])

    # start the interactive playthrough
    while True:
        try:
            for obs in map(render, ctrl):
                sleep(delay)

        except EOFError:
            break

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
        help="The stored replay data. If the file does not exist, enter"
             " freeplay mode, which records gameplay to the specified file.")

    parser.add_argument(
        '--delay', type=float, default=0.06, required=False, dest='delay',
        help="Delay between steps during replay.")

    parser.add_argument(
        '--seed', required=False, type=int, nargs=2,
        help="the seed pair to use, when free playing to a file.")

    parser.add_argument(
        '--debug', required=False, dest='debug', action='store_true',
        help="Enter trace mode.")

    parser.set_defaults(delay=0.06, debug=False)

    args, _ = parser.parse_known_args()
    replay(**vars(args))
