from collections import deque
from collections.abc import Iterable

from inspect import isgenerator, getgeneratorlocals
from nle_toolbox.bot.genfun import is_suspended

from gym import Wrapper, ActionWrapper


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


class BaseTemporalAbstractionWrapper(Wrapper):
    """Temporal Abstraction Wrapper

    Parameters
    ----------
    env : gym.Env
        The environment which to apply temporal abstraction to.

    Details
    -------
    Derived classes must implement an `.action` method, which returns
    a policy for the specified action. A policy is implemented as a generator
    that communicates low-level actions to the wrapped environment via `yield`
    statements like this:

    ```python
        def policy(obs, hx=None):
            action, done, hx = core(obs, hx=hx)
            while not done:  # note that the core decides when to cease
                obs, rew, _, info = yield action
                action, done, hx = core(obs, hx=hx)
    ```
    """
    _generator = None

    def action(self, action, observation):
        raise NotImplementedError

    def reset(self):
        """Reset the environment's state and return an initial observation."""
        if isgenerator(self._generator):
            self._generator.close()

        self._generator = self.loop(self.env)

        # no need to catch StopIteration, since by design our generator is
        # non-empty, because gym.Env's episodes cannot terminate during reset.
        obs, rew, done, info = self._generator.send(None)

        return obs

    def step(self, action):
        """Run one abstract timestep of the environment's dynamics."""
        try:
            # peek into generator's context and get the observation it is
            # currently suspended at.
            obs = getgeneratorlocals(self._generator).get('obs')
            # XXX This feels like an abstraction leak even though we access
            # it in the read-only manner.
            return self._generator.send(self.action(action, obs))

        except RuntimeError:
            raise RuntimeError(f"Bad action `{action}`") from None

        except StopIteration as e:
            return e.value

    def close(self):
        """Perform the necessary cleanup."""
        if isgenerator(self._generator):
            self._generator.close()

        return super().close()

    @staticmethod
    def loop(env, *, reduce=sum):
        """The core temporal abstraction loop.

        Parameters
        ----------
        env : gym.Env
            The environment which to apply temporal abstraction to.

        reduce : callable, default=sum
            The callable which takes a list of rewards (floats or arrays, if
            the env has vector rewards) and computes and aggregate reward.
            By default we sum whatever rewards were tracked.

        Details
        -------
        It resets the environment, and then yields 4-tuples returned by the
        .step of the env, except that the rewards are aggregated since the last
        yield.
        """
        obs, rewards, done, info = env.reset(), [0.], False, {}
        while not done:
            # request the next policy from the downstream
            gen = yield obs, reduce(rewards), False, info
            # XXX we do not have a dedicated `.startup` sequence, which could
            # perform necessary in-env preparations, inits and configs, since
            # the first `.step` can set things up.
            try:
                rewards = []
                act = gen.send(None)
                while True:
                    # step and accumulate the rewards for the downstream consumer
                    obs, rew, done, info = env.step(act)
                    rewards.append(rew)
                    if done:
                        break

                    # get the next action
                    act = gen.send((obs, rew, done, info))

            except StopIteration:
                # forbid empty generators
                if not rewards:
                    raise RuntimeError

        # communicate through `StopIteration` whatever the last response from
        # the env was.
        # XXX Recall that a generator cannot raise StopIteration explicitly
        # (or bubble it from sub-generators). The exception can be throwns
        # only implicitly via `return`.
        return obs, reduce(rewards), done, info


class Continue(Exception):
    """Special action to continue with the preempted option."""
    def __init__(self, *args, **kwargs):
        super().__init__()


class Interject(Exception):
    """Special action to temporarily suspend the current option and
    give control to another one.
    """
    def __init__(self, option, *args, **kwargs):
        super().__init__(option)


class AtomicOptionWrapper(Wrapper):
    """Temporal Abstraction Wrapper with interruptible atomic composite action.

    Parameters
    ----------
    env : gym.Env
        The environment which to apply temporal abstraction to.

    reduce : callable, default=sum
        A callable, which aggregates the rewards given to it in a list.

    allow_empty : bool, default=False
        Whether to allow empty options, i.e. those which refuse to execute
        any actions even on initialization.

    Details
    -------
    The logic of the option interruption resembles the high-level design
    considered in

        Sutton, R.S., Precup, D., Singh, S. (1999) `A framework for
        temporal abstraction in reinforcement learning`, Artificial
        Intelligence 112, 181–211
    """
    _generator = None
    def __init__(self, env, *, reduce=sum, allow_empty=False):
        super().__init__(env)
        self.reduce = reduce
        self.allow_empty = allow_empty

    def reset(self):
        """Reset the environment's state and return an initial observation."""
        if isgenerator(self._generator):
            self._generator.close()

        self._generator = self.loop(
            self.env,
            reduce=self.reduce,
            allow_empty=self.allow_empty,
        )

        # no need to catch StopIteration, since by design our generator is
        # non-empty, because gym.Env's episodes cannot terminate during reset.
        obs, rew, done, info = self._generator.send(None)
        return obs

    def step(self, option=Continue):
        """Run one abstract time step of the environment's dynamics."""
        if not isgenerator(self._generator):
            raise RuntimeError("Please call `.reset` before stepping.")

        try:
            if not (
                isinstance(option, Exception) or
                isinstance(option, type) and issubclass(option, Exception)
            ):
                return self._generator.send(option)

            # try not to protect the generator from getting killed accidentally
            if isinstance(option, (Continue, Interject)) and not self.is_running:
                raise RuntimeError("No option is currently running.")

            # why forbid sending exceptions into the running loop?
            return self._generator.throw(option)

        except StopIteration as e:
            return e.value

    def close(self):
        """Perform the necessary cleanup."""
        if isgenerator(self._generator):
            self._generator.close()

        return super().close()

    @property
    def is_running(self):
        if not isgenerator(self._generator):
            return False

        # peek into generator's context and get the special flag
        ctx = getgeneratorlocals(self._generator)
        return ctx.get('is_running', False)

    @staticmethod
    def loop(env, *, reduce=sum, allow_empty=True):
        """Interact with the env in atomic action sequences, actively pooling
        for preemption by a downstream users.

        Parameters
        ----------
        env : gym.Env
            The environment which to apply temporal abstraction to.

        reduce : callable, default=sum
            The callable which takes a list of rewards (floats or arrays, if
            the env has vector rewards) and computes and aggregate reward.
            By default we sum whatever rewards were tracked.

        Details
        -------
        It resets the environment, and then yields 4-tuples returned by the
        .step of the env, except that the rewards are aggregated since the last
        yield.

        Design Considerations
        ---------------------
        A neat idea is to use `.step`-granular timer to schedule the next
        preemption by the manger (the downstream user), which checks if the
        control by the currently running option should not be overtaken by
        another option. This requires the user to communicate a parameter
        `n_slice`, which specifies the number of steps until the next
        preemption event.

        The complications came from deciding what determines this time-slice
        value. On the one hand, the manger, being the one to decide what and
        when to run should be able to set the time-slice. This protects against
        malicious/faulty options, whose policy sets the slice so high, as to
        effectively make it never yield the control flow. On the other hand,
        the option's policy's internal logic `knows` better how often it can
        be polled for interruption.

        From responsibility separation standpoints, the manager could not know
        the allowed preemption interval better than the option. Besides, since
        the manager issues policies that come from a trusted source, it might
        be better to assume cooperation between policies, meaning that a policy
        yields back control as soon as exclusive control becomes non-critical.

        Long before considering the interruptions it was decided that options'
        policies may communicate simple and composite actions, in the form of
        short sequences of instructions that are to be executed atomically,
        i.e. in an open-loop control fashion without exchanging responses.
        Since the policy decides the atomicity of the instructions, it seems
        reasonable to postpone preemption events until the current action has
        been executed in its entirety. If an option is low priority, i.e.
        considers preemption permissible and highly likely, then it can enable
        finest-grained control by issuing singleton actions. Otherwise, the
        instruction sequence determines the minimally required time slice.
        """
        obs, macro, done, info = env.reset(), [0.], False, {}

        # pipe0 is the low-level instruction execution pipeline
        option, micro, pipe0, stack = None, [], deque([]), []
        while not done:
            is_running = option is not None and is_suspended(option)

            try:
                # forbid preempting an already interjecting option
                if not stack:
                    # get the option to execute and reset the reward accumulator
                    #  programmatically the option is a generator, which yields
                    #  actions and receives the observations from the resulting
                    #  transitions:  s_t, a_t -->> s_{t+1}, r_{t+1}.
                    option = yield obs, reduce(macro), False, info
                    macro = []

                    # start by sending a `None` to the generator
                    #  see pytorch PR#49017 and PEP-342
                    pipe0.extend(option.send(None))

            except StopIteration:
                if not allow_empty:
                    # XXX it is still unclear what should we do with empty options
                    raise RuntimeError("Cannot execute empty options.") from None

            except Continue:
                # we definitely cannot continue something, that has not been
                #  started
                if not is_running:
                    raise RuntimeError(
                        "Cannot continue when no option is running."
                    ) from None

            except Interject as e:
                # suspend the currently running option
                stack.append(option)

                # read off the interjecting option, but do not reset the reward
                # accumulator
                option, = e.args
                pipe0.extend(option.send(None))

            try:
                micro = []
                # a non-preemptible open-loop sub-policy
                while pipe0 and not done:
                    # accumulate rewards from executing low-level instructions
                    obs, rew, done, info = env.step(pipe0.popleft())
                    micro.append(rew)

                # record the rewards for the manager
                macro.extend(micro)

                # fetch the next composite action
                if not pipe0:
                    pipe0.extend(option.send((obs, reduce(micro), done, info)))

            except StopIteration:
                # could not fetch a sequence of instructions, since the option
                #  has terminated

                if stack:
                    # pop any interjected options
                    option = stack.pop()

        # communicate the last observation through `StopIteration`
        return obs, reduce(macro), True, info
