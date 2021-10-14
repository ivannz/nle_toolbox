from inspect import isgenerator

from gym import Wrapper


def interact(env, *, can, handle):
    """A simple semi-automatic interaction loop. See `doc/shell`."""
    # construct the initial result
    obs, rew, done, info = env.reset(), 0., False, {}
    while not done:
        # if we cannot override then yield to the user
        if not can(obs, info=info):
            act = yield obs, rew, done, info

        else:
            # handle stuff
            act = handle(obs, rew, done, info)

        # interact with the environment
        obs, rew, done, info = env.step(act)

    return obs, rew, done, info


class BaseOverrider(Wrapper):
    """Gym interface for overrider."""
    _loop = None

    def reset(self, **kwargs):
        if isgenerator(self._loop):
            self._loop.close()

        # reset the generator and start it
        self._loop = interact(
            self.env,
            can=self.can,
            handle=self.handle,
        )

        obs, rew, done, info = self._loop.send(None)
        return obs

    def step(self, act):
        return self._loop.send(act)

    def can(self, obs, rew=0., done=False, info=None):
        return False

    def handle(self, obs, rew=0., done=False, info=None):
        raise NotImplementedError
