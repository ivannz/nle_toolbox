import numpy as np

import gym
import gym_minigrid  # noqa: F401


DEFAULT_MINIGRID_NAVIGATION_ACTIONS = (
    "left",
    "right",
    "forward",
    "toggle",  # open doors, interact with objects
)


class MinigridNavigationWrapper(gym.ActionWrapper):
    """Limit the actions sapce to the specified human-readable actions.

    Warning
    -------
    The actions are numbered according to the order provided at init.
    """

    from gym_minigrid.minigrid import MiniGridEnv

    action_names = {v.name: int(v) for v in MiniGridEnv.Actions}

    def __new__(cls, env, actions=None):
        from collections import Counter

        if actions is None or not actions:
            return env

        absent = [n not in cls.action_names for n in actions]
        if any(absent):
            raise ValueError(f"unrecognized actions `{absent}`.")

        ctr = Counter(actions)
        dup = [n for n, v in ctr.items() if v > 1]
        if dup:
            raise ValueError(f"duplicate actions `{dup}`.")

        return object.__new__(cls)

    def __init__(self, env, actions=None):
        super().__init__(env)

        # create forward and reverse mappings
        self.forward = tuple(map(self.action_names.get, actions))
        self.reverse = dict(zip(self.forward, range(len(actions))))
        self.action_space = gym.spaces.Discrete(len(self.forward))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def action(self, action):
        return self.forward[action]

    def reverse_action(self, action):
        return self.reverse[action]


class RandomInstanceMinigird(gym.Env):
    """Randomize across different partially observed instances of the Minigrid."""

    def __init__(self, *specs, with_replacement=True):
        self.with_replacement = with_replacement
        self.specs = specs

        # assume the factory produces envs with identical observation and
        #  action spaces
        self.envs = tuple(map(gym.make, specs))
        self.seed(None)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def seed(self, seed=None):
        from sys import maxsize

        ss = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(ss)

        rngs = map(np.random.default_rng, ss.spawn(len(self.envs)))
        for env, rng in zip(self.envs, rngs):
            env.seed(int(rng.integers(maxsize)))

        return ss.entropy

    @property
    def id(self):
        if not hasattr(self, "current"):
            raise RuntimeError("Please call `.reset` first.")

        return self.specs[self.current]

    @property
    def env(self):
        if not hasattr(self, "current"):
            raise RuntimeError("Please call `.reset` first.")

        return self.envs[self.current]

    def reset(self, seed=None):
        n_envs = len(self.envs)
        if not hasattr(self, "curriculum_"):
            from collections import deque

            self.curriculum = deque([], n_envs)

        if not self.curriculum:
            # either draw an env at random, or ensure that no env is picked
            #  twice in a row
            if self.with_replacement:
                indices = self.rng.integers(n_envs, size=n_envs)
            else:
                indices = self.rng.permutation(n_envs)

            self.curriculum.extend(map(int, indices))

        self.current = self.curriculum.popleft()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode)


def tty_str(self):
    OBJECT_TO_STR = {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }

    COLOR_TO_CL = {
        "black": 0,
        "red": 1,
        "green": 2,
        "yellow": 3,
        "blue": 4,
        "purple": 5,
        "cyan": 6,
        "grey": 7,
    }

    AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    TRBL_TO_WALL = {
        # trbl
        0b0000: "+",  # ....
        0b0001: "-",  # ...l
        0b0010: "|",  # ..b.
        0b0011: "+",  # ..bl
        0b0100: "-",  # .r..
        0b0101: "-",  # .r.l
        0b0110: "+",  # .rb.
        0b0111: "+",  # .rbl
        0b1000: "|",  # t...
        0b1001: "+",  # t..l
        0b1010: "|",  # t.b.
        0b1011: "+",  # t.bl
        0b1100: "+",  # tr..
        0b1101: "+",  # tr.l
        0b1110: "+",  # trb.
        0b1111: "+",  # trbl
    }

    H, W = self.grid.height, self.grid.width

    # border with SPACE for wrapping
    chars = np.full((H + 1, W + 1), 0x20, np.uint8)
    colors = np.full((H, W), 0, np.uint8)
    for j in range(colors.shape[0]):
        for i in range(colors.shape[1]):
            if i == self.agent_pos[0] and j == self.agent_pos[1]:
                ch = AGENT_DIR_TO_STR[self.agent_dir]
                col = "red"
            else:
                c = self.grid.get(i, j)
                if c is None or c.type == "floor":
                    ch, col = " ", "black"

                elif c.type == "door":
                    ch, col = "D", c.color
                    if c.is_open:
                        ch = "_"
                    elif c.is_locked:
                        ch = "L"

                else:
                    ch, col = OBJECT_TO_STR[c.type], c.color

            chars[j, i] = ord(ch)
            colors[j, i] = COLOR_TO_CL[col]

    # prettify connecting walls and doors
    is_wall = chars == ord("W")
    is_door = np.isin(chars, (ord("D"), ord("L"), ord("_")))
    is_wall_or_door = is_wall | is_door
    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            if is_wall[i, j]:
                chars[i, j] = ord(
                    TRBL_TO_WALL[
                        (
                            0
                            | (is_wall_or_door[i - 1, j] << 3)  # top
                            | (is_wall_or_door[i, j + 1] << 2)  # right
                            | (is_wall_or_door[i + 1, j] << 1)  # bottom
                            | (is_wall_or_door[i, j - 1] << 0)  # left
                        )
                    ]
                )

    # render ansi color escapes
    ansi = ""
    for r in range(colors.shape[0]):
        row = ""
        for c in range(colors.shape[1]):
            ch, cl = chars[r, c], colors[r, c]
            if cl:
                row += f"\033[38;5;{cl&0x7f:d}m"

            row += chr(ch)

        ansi += row + "\n"

    return ansi


def patch_MiniGridEnv_dunder_str():
    import importlib

    minigrid = importlib.import_module("gym_minigrid.minigrid")
    minigrid.MiniGridEnv.__repr__ = tty_str
    minigrid.MiniGridEnv.__str__ = tty_str


def render(self, tile_size, agent_pos=None, agent_dir=None, highlight_mask=None):
    img = self.render_original(
        tile_size,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        highlight_mask=highlight_mask,
    )

    if highlight_mask is not None:
        if not hasattr(self, "persistent_mask"):
            self.persistent_mask = np.ones_like(highlight_mask, float)

        # fade and reset visible to one
        # XXX lerp(a, b, f) = (1-f) * a + f * b = a + f * (b - a)
        self.persistent_mask *= 0.75  # persistence factor `eta`
        self.persistent_mask[highlight_mask] = 1.0
        # self.persistent_mask += (1 - eta) * (highlight_mask - self.persistent_mask)

        # rearrange(img, "(H P) (W Q) C -> H W P Q C")
        ih, iw, ch = img.shape
        grid = img.reshape(
            ih // tile_size,
            tile_size,  # n_rows, th
            iw // tile_size,
            tile_size,  # n_cols, tw
            ch,  # ch
        ).transpose(
            0, 2, 1, 3, 4
        )  # -->> n_rows, n_cols, th, tw, ch

        # n_cols, n_rows -->> n_rows, n_cols, 1, 1, 1
        grid = (
            (
                grid
                * np.expand_dims(
                    self.persistent_mask.T,
                    axis=(-1, -2, -3),
                )
            )
            .clip(0, 255)
            .astype(np.uint8)
        )

        # import pdb ; pdb.set_trace()
        img = grid.transpose(0, 2, 1, 3, 4).reshape(img.shape).copy()

    return img


def patch_minigrid_render_fadetoblack():
    """Mysterious darkness shrouds over unseen places.

    Details
    -------
    Cosmetic patch for Minigrid RGB image renderer, which makes partial
    observability a little more obvious to humans.

    The tiles, that are no longer visible, gradually fade away to make an
    illusion of situational uncertainty ("fog of war", limited memory and
    finite attention).
    """
    import importlib

    minigrid = importlib.import_module("gym_minigrid.minigrid")
    minigrid.Grid.render_original = minigrid.Grid.render
    minigrid.Grid.render = render


class RandomMinigirdDojoTrain(RandomInstanceMinigird):
    def __init__(self):
        super().__init__(
            # "MiniGrid-Empty-8x8-v0",
            "MiniGrid-Dynamic-Obstacles-8x8-v0",
            "MiniGrid-MultiRoom-N2-S4-v0",
            "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-SimpleCrossingS9N2-v0",
            with_replacement=False,
        )


class RandomMinigirdDojoEval(RandomInstanceMinigird):
    def __init__(self):
        super().__init__(
            # # train
            # "MiniGrid-Empty-8x8-v0",
            # "MiniGrid-Dynamic-Obstacles-8x8-v0",
            # "MiniGrid-MultiRoom-N2-S4-v0",
            # "MiniGrid-LavaCrossingS9N1-v0",
            # "MiniGrid-SimpleCrossingS9N2-v0",
            # # test
            # "MiniGrid-Empty-5x5-v0",
            # "MiniGrid-Empty-Random-5x5-v0",
            # "MiniGrid-Empty-6x6-v0",
            "MiniGrid-Empty-Random-6x6-v0",
            # "MiniGrid-Empty-8x8-v0",
            "MiniGrid-Empty-16x16-v0",
            "MiniGrid-FourRooms-v0",
            # "MiniGrid-MultiRoom-N2-S4-v0",
            "MiniGrid-MultiRoom-N4-S5-v0",
            "MiniGrid-MultiRoom-N6-v0",
            "MiniGrid-DistShift1-v0",
            "MiniGrid-DistShift2-v0",
            "MiniGrid-LavaGapS5-v0",
            "MiniGrid-LavaGapS6-v0",
            "MiniGrid-LavaGapS7-v0",
            # "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-LavaCrossingS9N2-v0",
            "MiniGrid-LavaCrossingS9N3-v0",
            "MiniGrid-LavaCrossingS11N5-v0",
            "MiniGrid-SimpleCrossingS9N1-v0",
            # "MiniGrid-SimpleCrossingS9N2-v0",
            "MiniGrid-SimpleCrossingS9N3-v0",
            "MiniGrid-SimpleCrossingS11N5-v0",
            # "MiniGrid-Dynamic-Obstacles-5x5-v0",
            "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
            # "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
            # "MiniGrid-Dynamic-Obstacles-8x8-v0",
            "MiniGrid-Dynamic-Obstacles-16x16-v0",
            # # simple demo
            # "MiniGrid-MultiRoom-N4-S5-v0",
            # "MiniGrid-FourRooms-v0",
            # "MiniGrid-MultiRoom-N6-v0",
            # "MiniGrid-FourRooms-v0",
            with_replacement=False,
        )
