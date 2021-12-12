import os
import pickle
import sys

from gym import Wrapper

from ..utils import seeding
from ..utils.io import mkstemp


class Replay(Wrapper):
    """Wrapper for NLE Challenge env that enables deterministic replay.

    Parameters
    ----------
    env : nle.env.base.NLE
        An instance of the NetHack Learning Environment (w. gym interface).

    sticky : bool, default=False
        A flag indicating whether auto-generated seeds should be redrawn on
        each env reset. If True, then once generated seed is permanent and
        reused on every reset.

    Attributes
    ----------
    _seed_type : str
        The source of the seed. 'auto' means that the seeds were obtained in
        a deterministic way from the system's entropy. 'manual' implies that
        the entropy was supplied by the user via an explicit `.seed` call with
        non None `seed` argument.

    env : gym.Env
        The nle environment being wrapped.

    Motivation
    ----------
    We force-seed the main challenge task, since we are training an agent,
    and we also want to diagnose its failures and improve it on scenaria,
    which had undesirable outcomes. Also, please, see docs in `.reset` below.

    Warning
    -------
    Consult with the docstring `seeding.set_seed` about the effects of
    re-seeding/reusing the same env instance multiple times.
    """
    from ..__version__ import __version__
    _seed_type = 'auto'

    def __init__(self, env, *, sticky=False):
        if not seeding.is_seedable():
            raise RuntimeError("It appears that Nethack env has been compiled "
                               "without seeding support.")

        super().__init__(env)
        self.sticky = sticky

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
        """Load the state into NetHack.

        Parameters
        ----------
        state_dict: dict
            The Replay state dictionary, which contains the exact seeds
            for NetHack, and the sequence actions to trace in a seeded env.

        strict: bool, default=True
            Determines strictness of version compatibility check.

        Returns
        -------
        obs: object
            The last observation from NetHack resulting from replaying
            the recorded actions in it.
        """
        from packaging.version import Version

        ver = Version(state_dict['__version__'])

        # maintain compatibility with older versions
        if ver > Version(self.__version__):
            raise RuntimeError(f"Unsupported version `{ver}`.")

        # reseed and reset
        self.seed(seed=state_dict['seed'])
        obs, fin, j = self.reset(), False, 0

        # blindly replay the actions
        actions = state_dict['actions']
        while not fin and j < len(actions):
            obs, rew, fin, info = self.step(actions[j])
            j += 1

        # the final observation
        return obs

    @property
    def root(self):
        """Return the underlying c-level instance of NetHack."""
        if not hasattr(self, '_pynethack'):
            self._pynethack = seeding.pyroot(self.unwrapped)

        return self._pynethack

    def seed(self, seed=None):
        """Sets the seed for NLE's random number generator."""

        # draw entropy from the system if seed is None
        self._seed_type = 'auto' if seed is None else 'manual'

        # make sure the seed is a pair
        if not isinstance(seed, tuple):
            seed = seeding.generate(seed)

        # cache the seeds to deterministic reset and replay
        self._seed = core, disp = seed

        # XXX `get_seeds` returns the seeds used to fire up the prng
        # core, disp, ignore = self.root.get_seeds()

        return core, disp

    def reset(self, seed=None):
        """Reset the environment to an initial seeded state.

        Details
        -------
        The logic of `.reset` in this particular wrapper VIOLATES the spirit of
        the PRNG interplay and resetting in gym as of version `0.21`. Gym's API
        mandates that the resets not affect the environment's random number
        generator and ensure that the RANDOMNESS consumed by the env during
        each episode is statistically INDEPENDENT between calls to `.reset`.

        Instead, here we ENSURE that NLE's worldgen, monsters, effects and
        loot, that are typically stochastically generated from the PRNG, are
        instead produced DETERMINISTICALLY and are IDENTICAL between resets.
        """
        # get the cached seed, or generate a new unpredictable one, unless
        #  seeds were set manually by a call to `.seed` with non None arg.
        if not hasattr(self, '_seed'):
            # XXX if seed is None, then `_seed_type` becomes `auto`, and
            #  later resets land on the next branch if `sticky` if False
            self.seed(seed)

        elif self._seed_type == 'auto' and not self.sticky:
            self.seed(None)

        # `set_initial_seeds` has effect only until the next reset, at which
        #  new seeds are drawn from cNethack's internal random generator.
        self.root.set_initial_seeds(*self._seed, False)

        # clear the action history
        self._actions = []

        # now call the parent reset
        return self.env.reset()

    def step(self, act):
        """Run one timestep of the environment's dynamics."""

        # record the action and step
        self._actions.append(act)
        return super().step(act)

    def replay(self, actions, *, seed):
        """Deterministically replay the specified actions in the env.

        Parameters
        ----------
        actions: iterable
            A finite iterable of actions that implement the deterministic
            replay. Essentially an open loop policy, since it disregards
            the feedback.

        seed: object
            The entropy used to seed NetHack with. If None then entropy is
            drawn from the system pool.

        Yields
        ------
        obs: object
            The current observation x_t.

        act: object
            The action a_t taken in response to x_t.

        rew: float
            The reward r_{t+1} received for the (x_t, a_t) -->> x_{t+1}
            transition.

        obs_: object
            The next observation x_{t+1} due to taking a_t at x_t.

        info: dict
            The info dict associated with the t -->> t+1 transition.

        Details
        -------
        Upon raising StopIteration this generator returns the list of remaining
        actions in attribute `.value` of the exception object as per PEP-342
        and python docs (see pytorch PR#49017 for a detailed description).
        """
        self.seed(seed)

        # XXX by design `.replay` yields at least one SARS transition
        #  for a non-empty list of actions.
        obs, fin, j = self.reset(), False, 0
        if not actions:
            # yield a semi-invalid SARS if there are no actions to replay
            yield None, None, float('nan'), obs, {}
            return []

        while not fin and j < len(actions):
            # blindly follow the prescribed sequence
            act = actions[j]
            obs_, rew, fin, info = self.step(act)

            # yield a SARS transition
            yield obs, act, rew, obs_, info
            obs, j = obs_, j + 1

        # StopIteration's `.value` always contains the returned value
        return actions[j:]

    def render(self, mode='human', **kwargs):
        """Custom human mode renderer for NLE."""

        # override 'human' rendering only
        if mode != 'human':
            return super().render(mode=mode)

        # get the necessary data, and fail gracefully if it is unavailable.
        # XXX perhaps we should use `..utils.env.render` here
        obs, keys = self.env.last_observation, self.env._observation_keys
        try:
            tty_chars = obs[keys.index("tty_chars")]
            tty_colors = obs[keys.index("tty_colors")]

        except ValueError:
            return False

        # render NetHack output line by line with ANSI escapes
        ansi = ''
        height, width = tty_chars.shape
        for L in range(height):
            # position the cursor at (L, 4) with \033[<L>;<C>H
            ansi += f'\033[{L+4};4H'
            for C in range(width):
                # set fg color with \033[<bold?>;3<3-bit color>m
                cl, ch = tty_colors[L, C], tty_chars[L, C]
                ansi += f'\033[{bool(cl & 8):1d};3{cl & 7:1d}m{ch:c}'

        # save/restore the original cursor, and reset the color back to normal
        sys.stdout.write(f'\033[s{ansi}\033[u\033[m')
        sys.stdout.flush()

        return True


class ReplayToFile(Replay):
    """The Replay wrapper, which saves on `.reset` and terminal `.step`.

    Attributes
    ----------
    env : gym.Env
        The wrapped NetHack Learning Environment.

    folder : str
        The folder, which is created if absent, into which the interactive
        replays are dumped under random filenames.

    prefix : str, default=''
        The prefix of each replay dump.

    sticky : bool, default=False
        Whether to renew the auto-generated seeds from the system entropy pool
        on each `.reset`. This flag has any effect only when the seed have not
        been set but the user.

    save_on : str, list of str
        The list of events that trigger a playthrough save. Saves on episode
        termination (`done`), a reset (`reset`) and when the env is being
        closed (`close`).

    Attributes
    ----------
    filename : str
        The filename into which the replay has been saved the last time.
    """

    def __init__(
        self,
        env,
        *,
        folder,
        prefix='',
        sticky=False,
        save_on='done,reset,close'
    ):
        # parse save triggers
        if isinstance(save_on, str):
            save_on = save_on.split(',')

        events = {ev: ev in save_on for ev in (
            'done', 'reset', 'close',
        )}
        assert any(events.values())

        super().__init__(env, sticky=sticky)

        # make sure the dump folder exists
        self.folder, self.prefix = os.path.abspath(folder), prefix
        os.makedirs(self.folder, exist_ok=True)

        self.triggers = frozenset(events)

    def save(self):
        """Save the current replay under a random name."""

        # generate a new random opaque filename
        self.filename = mkstemp(dir=self.folder, suffix='.pkl',
                                prefix=self.prefix)

        # save into the current `.filename`
        pickle.dump(self.state_dict(), open(self.filename, 'wb'))

        return self.filename

    def step(self, act):
        obs, rew, fin, info = super().step(act)

        # save on episode end
        if fin and 'done' in self.triggers:
            self.save()

        return obs, rew, fin, info

    def reset(self, **kwargs):
        # if `_actions` are absent then we have not been reset yet and
        #  there is nothing to save.
        if hasattr(self, '_actions') and 'reset' in self.triggers:
            self.save()

        return super().reset(**kwargs)

    def close(self):
        if 'close' in self.triggers:
            self.save()

        super().close()
