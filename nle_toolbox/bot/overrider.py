from collections import deque
from collections.abc import Iterable

from gym import Wrapper, ActionWrapper
from inspect import isgenerator, getgeneratorlocals

from .genfun import is_suspended


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


class OptionWrapper(Wrapper):
    """Temporal Abstraction Wrapper with atomic composite actions.

    Parameters
    ----------
    env : gym.Env
        The environment which to apply temporal abstraction to.

    reduce : callable, default=sum
        The callable which takes a list of rewards (floats or arrays, if
        the env has vector rewards) and computes and aggregate reward.
        By default we sum whatever rewards were tracked.

    allow_empty : bool, default=False
        Whether to allow empty options, i.e. those which refuse to execute
        any actions even on initialization.

    Details
    -------
    Actions of this particular wrapper are policies, implemented as python
    generators, that communicate low-level simple of composite actions to
    the wrapped environment via `yield` statements like this:

    ```python
        def policy(obs, hx=None):
            # note that the core decides when to cease
            action, halt, hx = core(obs, hx=hx)
            while not halt:
                obs, rew, _, info = yield (action,)  # a singleton action
                # halting after `s_t, a_t -->> s_{t+1}` depends on `s_t`!
                action, halt, hx = core(obs, hx=hx)
    ```
    """
    _generator = None

    def __init__(self, env, *, reduce=sum, allow_empty=False):
        assert callable(reduce)

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

    def step(self, action):
        """Run one abstract time step of the environment's dynamics."""
        if not isgenerator(action):
            raise TypeError("The action must be a generator.")

        if not isgenerator(self._generator):
            raise RuntimeError("Please call `.reset` before stepping.")

        try:
            return self._generator.send(action)

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
    def loop(env, *, reduce=sum, allow_empty=False):
        """The core temporal abstraction loop.

        Parameters
        ----------
        env : gym.Env
            The environment which to apply temporal abstraction to.

        reduce : callable, default=sum
            The callable which takes a list of rewards (floats or arrays, if
            the env has vector rewards) and computes and aggregate reward.
            By default we sum whatever rewards were tracked.

        allow_empty : bool, default=False
            Whether to allow empty options, i.e. those which refuse to execute
            any actions even on initialization.

        Details
        -------
        It resets the environment, and then yields 4-tuples returned by the
        .step of the env, except that the rewards are aggregated since the last
        yield.
        """
        obs, rewards, done, info = env.reset(), [0.], False, {}

        option, pipe0 = None, deque([])
        while not done:
            is_running = option is not None and is_suspended(option)

            # request the next policy from the downstream
            option = yield obs, reduce(rewards), False, info
            rewards = []

            try:
                pipe0.extend(option.send(None))
                while not done:
                    micro = []
                    while pipe0 and not done:
                        obs, rew, done, info = env.step(pipe0.popleft())
                        micro.append(rew)

                    # accumulate the rewards for the downstream consumer
                    rewards.extend(micro)

                    if not pipe0:
                        pipe0.extend(option.send((obs, reduce(micro), done, info)))
                        micro = []

            except StopIteration:
                if allow_empty and not rewards:
                    raise RuntimeError(
                        "Cannot execute empty options."
                    ) from None

        # communicate the last response through `StopIteration`
        return obs, reduce(rewards), True, info


class Continue(Exception):
    """Special action to continue executing the preempted option."""
    def __init__(self, *args, **kwargs):
        super().__init__()


class Preempt(Exception):
    """Special action to temporarily suspend the current option and
    give control to another one.
    """
    def __init__(self, option, *args, **kwargs):
        super().__init__(option)


class InterruptibleOptionWrapper(OptionWrapper):
    """Temporal Abstraction Wrapper with interruptible composite actions.

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
            if isinstance(option, (Continue, Preempt)) and not self.is_running:
                raise RuntimeError("No option is currently running.")

            # why forbid sending exceptions into the running loop?
            return self._generator.throw(option)

        except StopIteration as e:
            return e.value

    @staticmethod
    def loop(env, *, reduce=sum, allow_empty=False):
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

        allow_empty : bool, default=False
            Whether to allow empty options, i.e. those which refuse to execute
            any actions even on initialization.

        Details
        -------
        It resets the environment, and then yields 4-tuples returned by the
        .step of the env, except that the rewards are aggregated since the last
        yield.

        Design Considerations
        ---------------------
        A neat idea is to use `.step`-granular timer to schedule the manager's
        (the downstream user) next inspection, during which it checks if the
        control currently held by the running option should be transferred to
        another option. This requires the user to communicate a parameter,
        which specifies the number of steps until the next event.

        The complications came from deciding what determines this timeout. On
        the one hand, the supervisor decides when and what to run, hence should
        be able to set the time slice. This protects against malicious/faulty
        options, whose policy ramps the slice so high as to effectively make it
        hold exclusive control. On the other hand, the option's policy's inner
        logic `knows` better how often it can be polled for inspection. In
        addition, from responsibility separation standpoint, the supervisor
        could not know the acceptable polling interval better than the option
        itself. Besides, since the manager takes policies from a trusted source,
        it might be reasonable to assume cooperation between policies, meaning
        that a policy yields back control as soon as exclusive control becomes
        non-critical. This leaves faulty policies, which for some reason fail
        to halt.

        Long before considering the interruptions it was decided that besides
        simple actions options' policies may communicate composite actions, i.e.
        short sequences of instructions that are to be executed consecutively
        disregarding feedback from the environment (open-loop control), e.g.
        atomically. Since the policy inherently decides the atomicity of its
        actions, it seems reasonable to postpone the supervisor's inspection
        events until the current action has been executed in its entirety. For
        example, if an option is low priority, i.e. allows interruption and
        considers it highly likely, then it can opt in for the finest-grained
        supervision by issuing simple actions (singleton).

        Thus it was decided that the instruction sequence of an action
        determines the number of steps until next inspection.
        """
        obs, macro, done, info = env.reset(), [], False, {}

        # pipe0 is the low-level instruction execution pipeline
        option, pipe0, stack = None, deque([]), []
        while not done:
            is_running = option is not None and is_suspended(option)

            try:
                # forbid interrupting an already preempted option, i.e. the new
                # options assumes exclusive control
                if not stack:
                    # get the option to execute and reset the reward accumulator
                    #  programmatically the option is a generator, which yields
                    #  actions and receives the observations from the resulting
                    #  transitions:  s_t, a_t -->> s_{t+1}, r_{t+1}.
                    option = yield obs, reduce(macro), False, info  # done
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

            except Preempt as e:
                # suspend the currently running option, start the preempting
                # option, but do not reset the reward accumulator
                stack.append(option)

                option, = e.args
                pipe0.extend(option.send(None))

            try:
                # atomically execute the composite action, i.e. an open-loop
                #  sub-policy that cannot be non-interrupted
                micro = []
                while pipe0 and not done:
                    # accumulate rewards from executing low-level instructions
                    obs, rew, done, info = env.step(pipe0.popleft())
                    micro.append(rew)

                # record the rewards for the supervisor
                macro.extend(micro)

                # fetch the next composite action
                if not pipe0:
                    pipe0.extend(option.send((obs, reduce(micro), done, info)))

            except StopIteration:
                # could not fetch a sequence of instructions, since the option
                #  has terminated
                if stack:
                    # pop the most recently preempted option
                    option = stack.pop()

        # communicate the last observation through `StopIteration`
        # XXX Recall that a generator cannot raise StopIteration explicitly
        # (or bubble it from sub-generators). The exception can be thrown only
        # implicitly via `return`.
        return obs, reduce(macro), True, info
