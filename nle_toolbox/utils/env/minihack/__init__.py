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

from nle import nethack
from pkg_resources import resource_filename

# the essential navigation actions and basic GUI handling
BASE_ACTIONS = (
    *nethack.CompassDirection,
    nethack.MiscDirection.WAIT,
    # these service actions are for GUI
    nethack.MiscAction.MORE,  # \015
    nethack.TextCharacters.SPACE,  # \040
    nethack.Command.ESC,  # \033
)

# extended, composite, or actions, which potentially summon a menu
EXT_ACTIONS = (
    *BASE_ACTIONS,
    nethack.Command.SEARCH,  # single-step search action
    nethack.Command.KICK,  # expects direction
    nethack.Command.EAT,  # might expect a choice form inv
)

CORRIDOR_MAP = """
-----       ----------------------
|...|       |....................|
|....#######.....................|
|...|       |....................|
-----       ----------------------
"""


class MiniHackFightCorridorDarkRandomRats(MiniHackNavigation):
    """A more rat infested dark corridor battle.

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

    def __init__(
        self,
        *args,
        # the max number of rats spawned at next to the exit
        n_giant_rats: int = 6,
        # whether the number of basic giant rats is fixed or random
        det: bool = True,
        # the number of rats spawned randomly around the map
        n_extra_rats: int = 0,
        # This env is unlit by default
        lit: bool = False,
        # Play with human knight character by default
        character: str = "kni-hum-law-fem",
        # Default episode limit
        max_episode_steps: int = 350,
        # remaining kwargs (see `MiniHackNavigation`)
        **other,
    ):
        # use `flags=('premapped',)` and `lit=True` to show the entre map
        lvl_gen = LevelGenerator(map=CORRIDOR_MAP, lit=False)
        lvl_gen.set_start_rect((1, 1), (3, 3))
        lvl_gen.add_goal_pos((32, 2))

        # the following script places a random number of rats in the original
        #  spawn rect. The infestation spills over the rect if overcrowrded.
        #  XXX NdD dice notation means draw N D-sided dice. We allow no rats!
        if det:
            n_rats_expr = str(n_giant_rats)
        else:
            n_rats_expr = f"1d{n_giant_rats + 1} + (-1)"
        lvl_gen.add_line(
            f"""
            $spawn = selection: fillrect(30, 1, 31, 3)
            $n_giant_rats = {n_rats_expr}
            IF [$n_giant_rats > 0] {{
                LOOP [$n_giant_rats] {{
                    MONSTER: "giant rat", rndcoord($spawn)
                }}
            }}
        """
        )

        # spawn certain extra rats at random locations
        for _ in range(n_extra_rats):
            lvl_gen.add_monster(name="giant rat", place="random")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            character=character,
            max_episode_steps=max_episode_steps,
            **other,
        )


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
    id="MiniHack-CorridorBattle-Dark-MoreRats-v0",
    entry_point=MiniHackFightCorridorDarkRandomRats,
    kwargs=dict(
        # -- More rats will change everything.
        # -- For the better, right?
        # -- ...
        # -- Right?
        n_giant_rats=6,
        det=True,
        n_extra_rats=6,
        reward_win=+1,
        reward_lose=-1,
    ),
)

register(
    id="MiniHack-CorridorBattle-Dark-RandomRats-v0",
    entry_point=MiniHackFightCorridorDarkRandomRats,
    kwargs=dict(
        n_giant_rats=10,  # 0-10 rats
        det=False,
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

register(
    id="MiniHack-CorridorBattle-Dark-MoreActions-v0",
    entry_point="minihack.envs.fightcorridor:MiniHackFightCorridorDark",
    kwargs=dict(
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)

register(
    id="MiniHack-Room-Ultimate-15x15-MoreActions-v0",
    entry_point="minihack.envs.room:MiniHackRoom15x15Ultimate",
    kwargs=dict(
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)

register(
    id="MiniHack-HideNSeek-Big-MoreActions-v0",
    entry_point="minihack.envs.hidenseek:MiniHackHideAndSeekBig",
    kwargs=dict(
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)

register(
    id="MiniHack-CorridorBattle-Dark-MoreRats-MoreActions-v0",
    entry_point=MiniHackFightCorridorDarkRandomRats,
    kwargs=dict(
        n_extra_rats=6,
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)

register(
    id="MiniHack-CorridorBattle-Dark-RandomRats-MoreActions-v0",
    entry_point=MiniHackFightCorridorDarkRandomRats,
    kwargs=dict(
        n_giant_rats=10,  # 0-10 rats
        det=False,
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)


class MiniHackFightCustomCorridorDark(MiniHackNavigation):
    def __init__(
        self,
        *args,
        # Play with human knight character by default
        character: str = "kni-hum-law-fem",
        # Default episode limit
        max_episode_steps: int = 500,
        # remaining kwargs (see `MiniHackNavigation`)
        **other,
    ):
        with open(resource_filename(__name__, "corridor.des"), "rt") as f:
            des_file = f.read()

        super().__init__(
            *args,
            des_file=des_file,
            character=character,
            max_episode_steps=max_episode_steps,
            **other,
        )


register(
    id="MiniHack-FightCustomCorridor-Dark-v0",
    entry_point=MiniHackFightCustomCorridorDark,
    kwargs=dict(
        reward_win=+1,
        reward_lose=-1,
        actions=BASE_ACTIONS,
    ),
)
