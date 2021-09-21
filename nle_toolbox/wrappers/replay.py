from gym import Wrapper

from ..utils import seeding


class Replay(Wrapper):
    """Wrapper for NLE Challenge env that enables deterministic replay.

    Motivation
    ----------
    We force-seed the main challenge task, since we are training an agent,
    and we also want to diagnose its failures and improve it on scenaria,
    which had undesirable outcomes.

    Warning
    -------
    Consult with the docstring `seeding.set_seed` about the effects of
    re-seeding/reusing the same env instance multiple times.
    """
    from ..__version__ import __version__

    def __init__(self, env):
        if not seeding.is_seedable():
            raise RuntimeError("It appears that Nethack env has been compiled "
                               "without seeding support.")

        super().__init__(env)

    def state_dict(self):
        """Get the replay state of the env for serialization."""
        from time import strftime

        return {
            '__version__': self.__version__,
            '__dttm__': strftime('%Y%m%d-%H%M%S'),
            'seed': getattr(self, '_seed', None),
            'actions': self._actions,
        }

    def load_state_dict(self, state_dict, *, strict=True):
        """Load the sate, reseed and replay."""

        from packaging.version import Version

        ver = Version(state_dict['__version__'])

        # maintain compatibility with older versions
        if ver > Version(self.__version__):
            raise RuntimeError(f"Unsupported version `{ver}`.")

        # replay the recorded actions
        return self.replay(state_dict['actions'], seed=state_dict['seed'])

    @property
    def root(self):
        if not hasattr(self, '_pynethack'):
            self._pynethack = seeding.pyroot(self.unwrapped)
        return self._pynethack

    def seed(self, seed=None):
        if not isinstance(seed, tuple):
            seed = seeding.generate(seed)

        # cache the seeds to deterministic reset and replay
        self._seed = core, disp = seed

        # XXX `get_seeds` returns the seeds used to fire up the prng
        # core, disp, ignore = self.root.get_seeds()

        return core, disp

    def reset(self, **kwargs):
        # get the cached seed, or generate a new unpredictable one.
        if not hasattr(self, '_seed'):
            self.seed(None)

        # `set_initial_seeds` has effect only until the next reset, at which
        #  new seeds are drawn from cNethack's internal random generator.
        self.root.set_initial_seeds(*self._seed, False)

        # clear the action history
        self._actions = []

        # now call the parent reset
        return super().reset(**kwargs)

    def step(self, act):
        # record the action and step
        self._actions.append(act)
        return super().step(act)

    def replay(self, actions, *, seed):
        """Deterministically replay the specified actions in the env."""

        self.seed(seed)

        obs, fin, j = self.reset(), False, 0

        history = [obs]
        while not fin and j < len(actions):
            obs, rew, fin, info = self.step(actions[j])
            history.append(obs)
            j += 1

        return history, actions[j:]
