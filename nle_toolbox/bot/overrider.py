from inspect import isgenerator
from nle_toolbox.bot.genfun import is_suspended

from gym import Wrapper


class UnhandledObservation(Exception):
    """Signal the caller that an Overrider cannot deal with the data."""
    pass


def follow(env, *, handle):
    """A simple semi-automatic interaction loop in follower mode.
    See `doc/shell`.
    """
    # construct the initial result
    obs, rew, done, info = env.reset(), 0., False, {}
    rew_lead = rew_follow = rew
    while not done:
        # if we cannot override then yield to the user
        # XXX `done` is always False inside the loop, so we can hardcode it
        try:
            act = handle(obs, rew_lead, False, info)
            from_lead = True  # set the flag after, since peer may raise

        except UnhandledObservation:
            # we implicitly assume that the may user throw an unhandled at us,
            # but the roles of the user and the peer may be reversed!
            act = yield obs, rew_follow, False, info
            from_lead = False
            # XXX whoever's here is not the peer. more like an advisor. So
            #  what credit does the advisor get?

        # interact with the environment
        obs, rew, done, info = env.step(act)

        # The reward is due to the transition induced by most recent action.
        #  Therefore we must assign credit to the correct decision maker.
        if from_lead:
            rew_lead = rew

        else:
            rew_follow = rew

    # the peer does not get to experience the end-of-episode signal,
    # so always send the final result with the user's reward.
    # XXX `done` is always True here
    return obs, rew_follow, True, info


def lead(env, *, handle):
    """A simple semi-automatic interaction loop in leader mode.
    See `doc/shell`.

    Details
    -------
    The same logic as in `follow`, but with yield and `handle` swapped places.
    """
    obs, rew, done, info = env.reset(), 0., False, {}
    rew_lead = rew_follow = rew
    while not done:
        try:
            act = yield obs, rew_follow, False, info
            from_lead = True

        except UnhandledObservation:
            act = handle(obs, rew_lead, False, info)
            from_lead = False

        obs, rew, done, info = env.step(act)

        if from_lead:
            rew_lead = rew

        else:
            rew_follow = rew

    return obs, rew_lead, True, info


def drive(env, *, device, reduce=sum):
    """A simple semi-automatic interaction loop in driver mode.
    See `doc/shell`.

    Existential
    -----------
    We assume too much about the env here, MB it would be better just to make
    an env class with the special, high-level action built in it `.step`,
    with the init in its `.reset`.
    """

    # construct the initial result
    tx = obs, rew, done, info = env.reset(), 0., False, {}

    # `None` command indicates startup
    gen, n_yields, n_steps, rewards = device(*tx, cmd=None), 0, 0, []
    # XXX tracking rewards in a list may consume a lot of mem!
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

            rew = reduce(rewards, start=0.)
            gen = device(*tx, cmd=(yield obs, rew, False, info))

            n_yields, n_steps, rewards = n_yields + 1, 0, []
            continue

        tx = obs, rew, done, info = env.step(act)
        rewards.append(rew)
        n_steps += 1

    # `done=True` is important enough to bubble upstream
    return obs, rew, True, info


class BaseOverrider(Wrapper):
    """Gym interface for overrider."""
    _loop = None

    def __init__(self, env, *, mode='driver'):
        assert mode in ('leader', 'follower', 'driver')
        super().__init__(env)

        self.mode = mode

    def reset(self, **kwargs):
        if isgenerator(self._loop):
            self._loop.close()

        # reset the generator and start it
        if self.mode == 'driver':
            self._loop = drive(self.env, device=self.handle)

        elif self.mode == 'leader':
            self._loop = lead(self.env, handle=self.handle)

        else:
            self._loop = follow(self.env, handle=self.handle)

        try:
            obs, rew, done, info = self._loop.send(None)
            return obs

        except StopIteration:
            raise RuntimeError("Terminated environment on reset!") from None

    def abstain(self):
        if not self.mode == 'lead':
            raise RuntimeError('Abstention is allowed only in leader mode.')

        try:
            return self._loop.throw(UnhandledObservation)

        except StopIteration as e:
            return e.value

    def step(self, act):
        try:
            return self._loop.send(act)

        except StopIteration as e:
            return e.value

    def handle(self, obs, rew=0., done=False, info=None, cmd=None):
        """Throwing an exception allows partial handling: we can update
        internal state and data, yet still yield to the user."""
        raise UnhandledObservation
