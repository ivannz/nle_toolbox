"""The original tasks from Minihack emit zero reward for death, which makes
this end-of-episode event indistinguishable from the time-out. See the base
class

    [MiniHack](https://github.com/facebookresearch/minihack/blob/65fc16f0f321b00552ca37db8e5f850cbd369ae5/minihack/base.py#L131-L132)

which is subclassed by `MiniHackNavigation` and `MiniHackRoom`. At the same
time the environments issue a slightly negative reward for bumping into walls.

This makes sense if we consider `death` itself as a meta-learning signal, which
affects somehow agent's learning process between episodes, not via a reward
within an episode.

In order to simplify the experimentation and not get distracted by intriducing
survival/selection elements into the RL training algorithm, we register new
envs with reshaped rewards, which punish death end-of-episode events by a `-1`
reward. This forces the agent to learn to avoid death and survive early on, and
then focus on actually seeking the positive reward for task completion.
"""

from minihack import MiniHackNavigation, LevelGenerator
from minihack.envs import register


class MiniHackFightCorridorDarkMoreRats(MiniHackNavigation):
    """A more ratty dark corridor battle.

    'Rats' by Jerma
    ---------------
    [Verse: Jerma]
    Rats, rats, we're the rats
    We prey at night, we stalk at night, we're the rats

    [King Rat]
    I'm the giant rat that makes all of the rules

    [All Rats]
    Let's see what kind of trouble we can get ourselves into
    """
    def __init__(self, *args, n_extra_rats=0, lit=False, **kwargs):
        kwargs["character"] = "kni-hum-law-fem"  # tested on human knight
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 350)

        map = """
-----       ----------------------
|...|       |....................|
|....#######.....................|
|...|       |....................|
-----       ----------------------
"""
        lvl_gen = LevelGenerator(map=map, lit=lit)
        lvl_gen.set_start_rect((1, 1), (3, 3))
        lvl_gen.add_monster(name="giant rat", place=(30, 1))
        lvl_gen.add_monster(name="giant rat", place=(30, 2))
        lvl_gen.add_monster(name="giant rat", place=(30, 3))
        lvl_gen.add_monster(name="giant rat", place=(31, 1))
        lvl_gen.add_monster(name="giant rat", place=(31, 2))
        lvl_gen.add_monster(name="giant rat", place=(31, 3))
        lvl_gen.add_goal_pos((32, 2))

        for _ in range(n_extra_rats):
            lvl_gen.add_monster(name="giant rat", place="random")

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)


register(
    id="MiniHack-CorridorBattle-Dark-v1",
    entry_point="minihack.envs.fightcorridor:MiniHackFightCorridorDark",
    kwargs=dict(
        reward_win=+1,  # default value as in the base class
        reward_lose=-1,  # <<-- reward for death, used to be 0.
    ),
)

# A dark corridor battle with six additional randomly placed rats.
# XXX We have observed that the difficulty spike diminishes with more rats.
register(
    id='MiniHack-CorridorBattle-Dark-MoreRats-v0',
    entry_point=MiniHackFightCorridorDarkMoreRats,
    kwargs=dict(
        # -- More rats will change everything.
        # -- For the better, right?
        # -- ...
        # -- Right?
        n_extra_rats=6,
        reward_win=+1,
        reward_lose=-1,
    ),
)

register(
    id="MiniHack-Room-Ultimate-15x15-v1",
    entry_point="minihack.envs.room:MiniHackRoom15x15Ultimate",
    kwargs=dict(reward_win=+1, reward_lose=-1),
)

register(
    id="MiniHack-HideNSeek-Big-v1",
    entry_point="minihack.envs.hidenseek:MiniHackHideAndSeekBig",
    kwargs=dict(reward_win=+1, reward_lose=-1),
)

register(
    id="MiniHack-Memento-F4-v1",
    entry_point="minihack.envs.memento:MiniHackMementoF4",
    kwargs=dict(reward_win=+1, reward_lose=-1),
)
