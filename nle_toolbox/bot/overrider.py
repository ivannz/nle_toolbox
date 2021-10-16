from inspect import isgenerator
from nle_toolbox.bot.genfun import is_suspended

from gym import Wrapper


class UnhandledObservation(Exception):
    """Signal the caller that an Overrider cannot deal with the data."""
    pass


def interact(env, *, peer):
    """A simple semi-automatic interaction loop in peer mode. See `doc/shell`.
    """
    # construct the initial result
    obs, rew, done, info = env.reset(), 0., False, {}
    while not done:
        # if we cannot override then yield to the user
        # XXX `done` is always False inside the loop, so we can hardcode it
        try:
            act = peer(obs, rew, False, info)

        except UnhandledObservation:
            act = yield obs, rew, False, info

        # interact with the environment
        obs, rew, done, info = env.step(act)

    # XXX `done` is always True here
    return obs, rew, True, info


def operate(env, *, follower):
    """A simple semi-automatic interaction loop in follower mode.
    See `doc/shell`.
    """
    # construct the initial result
    tx = obs, rew, done, info = env.reset(), 0., False, {}

    # `None` command indicates startup
    gen, n_yields, n_steps = follower(*tx, cmd=None), 0, 0
    while not done:
        try:
            act = gen.send(tx if is_suspended(gen) else None)

        except StopIteration:
            # We have initialized the generator with the same `tx` data.
            # Therefore we halt, if we get two immediate `StopIteration`-s
            # in a row, i.e. without any intermediate `.step`-s.
            if n_yields > 0 and n_steps < 1:
                # XXX maybe raise a warning instead?
                raise RuntimeError

            gen = follower(*tx, cmd=(yield obs, rew, False, info))
            n_yields += 1
            continue

        tx = obs, rew, done, info = env.step(act)
        n_steps += 1

    # `done=True` is important enough to bubble upstream
    return obs, rew, True, info


class BaseOverrider(Wrapper):
    """Gym interface for overrider."""
    _loop = None

    def __init__(self, env, *, mode='peer'):
        assert mode in ('peer', 'follow')
        super().__init__(env)

        self.is_follow = mode == 'follow'

    def reset(self, **kwargs):
        if isgenerator(self._loop):
            self._loop.close()

        # reset the generator and start it
        if self.is_follow:
            self._loop = operate(self.env, follower=self.handle)
        else:
            self._loop = interact(self.env, peer=self.handle)

        try:
            obs, rew, done, info = self._loop.send(None)
            return obs

        except StopIteration:
            raise RuntimeError("Terminated environment on reset!") from None

    def step(self, act):
        try:
            return self._loop.send(act)

        except StopIteration as e:
            return e.value

    def handle(self, obs, rew=0., done=False, info=None, cmd=None):
        """Throwing an exception allows partial handling: we can update
        internal state and data, yet still yield to the user."""
        raise UnhandledObservation
