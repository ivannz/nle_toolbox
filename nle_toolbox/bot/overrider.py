from inspect import isgenerator

from gym import Wrapper


class UnhandledObservation(Exception):
    """Signal the caller that an Overrider cannot deal with the data."""
    pass


def interact(env, *, handle):
    """A simple semi-automatic interaction loop. See `doc/shell`."""
    # construct the initial result
    obs, rew, done, info = env.reset(), 0., False, {}
    while not done:
        # if we cannot override then yield to the user
        # XXX `done` is always False inside the loop, so we can hardcode it
        try:
            act = handle(obs, rew, False, info)

        except UnhandledObservation:
            act = yield obs, rew, False, info

        # interact with the environment
        obs, rew, done, info = env.step(act)

    # XXX `done` is always True here
    return obs, rew, True, info


class BaseOverrider(Wrapper):
    """Gym interface for overrider."""
    _loop = None

    def reset(self, **kwargs):
        if isgenerator(self._loop):
            self._loop.close()

        # reset the generator and start it
        self._loop = interact(self.env, handle=self.handle)
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

    def handle(self, obs, rew=0., done=False, info=None):
        """Throwing an exception allows partial handling: we can update
        internal state and data, yet still yield to the user."""
        raise UnhandledObservation
