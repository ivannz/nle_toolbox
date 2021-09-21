import os
import pickle

from gym import Wrapper

from ..utils import seeding
from ..utils.io import mkstemp


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
            'actions': getattr(self, '_actions', []),
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


class ReplayToFile(Replay):
    """The Replay wrapper, which saves on `.reset` and terminal `.step`.

    Attributes
    ----------
    filename : str
        The filename into which the interactive replay is to be saved.
    """

    def __init__(self, env, *, folder, prefix=''):
        super().__init__(env)

        # make sure the dump folder exists
        self.folder, self.prefix = os.path.abspath(folder), prefix
        os.makedirs(self.folder, exist_ok=True)

    def save(self):
        # generate a new random opaque filename
        self.filename = mkstemp(dir=self.folder, suffix='.pkl',
                                prefix=self.prefix)

        # save into the current `.filename`
        pickle.dump(self.state_dict(), open(self.filename, 'wb'))

        return self.filename

    def step(self, act):
        obs, rew, fin, info = super().step(act)

        # save on episode end
        if fin:
            self.save()

        return obs, rew, fin, info

    def reset(self, **kwargs):
        # if `_actions` are absent then we have not been reset yet and
        #  there is nothing to save.
        if hasattr(self, '_actions'):
            self.save()

        return super().reset(**kwargs)
