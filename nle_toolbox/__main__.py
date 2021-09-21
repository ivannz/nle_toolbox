import re
from io import StringIO
from contextlib import redirect_stdout


import gym
import nle
from nle.nethack.actions import Command, MiscAction

from collections import deque

from nle_toolbox.bot.skeleton import Skeleton
from nle_toolbox.wrappers.replay import ReplayToFile, Replay
from nle_toolbox.utils.obs import BLStats, uint8_to_str


def render(env, obs):
    blstats = BLStats(*obs['blstats'])

    with redirect_stdout(StringIO()) as f:
        env.render('human')

    screen = f.getvalue()
    screen += "\n" + (b''.join(obs['inv_letters'].view('c'))).decode()

    screen += f"\ntime: {blstats.time:04d}"
    screen += "\n" + str(obs['message'].view('S256')[0])

    screen += "\n" + str(obs['glyphs'][
        blstats.y-2:blstats.y+3,
        blstats.x-2:blstats.x+3,
    ])

    print(screen, flush=True)
    return True


class HumanBrain:
    """A simple human actor that supports escaped input."""
    def override(self, obs):
        return False

    # human control
    def reset(self, obs):
        pass

    def step(self, obs):
        input = bytes(__builtins__.input('> '), 'utf8')
        yield from map(ord, input.decode('unicode-escape'))


if __name__ == '__main__':
    debug = False

    env = ReplayToFile(
        gym.make(
            'NetHackChallenge-v0',
            # XXX options affect even seeded generations, so we should prolly
            #  save them in the state.
            # options=(
            #     # 'color',
            #     # 'showexp',
            #     # 'nobones',
            #     # 'nolegacy',
            #     # 'nocmdassist',
            #     # 'disclose:+i +a +v +g +c +o',
            #     # 'runmode:teleport',
            #     # 'mention_walls',
            #     # 'nosparkle',
            #     # 'showscore',
            #     # 'pettype:none',
            # ),
        ), folder='./replays')

    # XXX in case we want to try out different scenaria see how a map is made
    #  in `minihack.envs.fightcorridor.MiniHackFightCorridor`.
    # from minihack.base import MiniHack
    # from minihack.level_generator import LevelGenerator
    # lvl_gen = LevelGenerator(map=..., lit=True)
    # MiniHack._patch_nhdat(env.unwrapped, lvl_gen.get_des())

    # Agent-Val-Hum-Fem-Law, can dual-wield!
    # env.seed(seed=(14278027783296323177, 11038440290352864458))

    # Agent-Pri-Elf-Fem-Cha can find lots of spells
    env.seed(seed=(5009195464289726085, 12625175316870653325))

    bot = Skeleton(brain=HumanBrain())

    while True:
        obs, rew, fin, info = env.reset(), 0., False, None
        bot.reset(obs)

        # nle/base.py#L382 maps gym action numbers to chars
        ctoa = {a: j for j, a in enumerate(env.unwrapped._actions)}
        while render(env, obs) and not fin:
            try:
                obs, rew, fin, info = env.step(ctoa[bot.step(obs)])

            except KeyError:
                if input('Invalid action. abort? [yn] (y)') != 'n':
                    break

                if debug:
                    breakpoint()

        # an extra prompt to break out from the loop
        if input('restart? [yn] (n)') != 'y':
            break
