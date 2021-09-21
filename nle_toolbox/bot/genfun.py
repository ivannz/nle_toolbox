from sys import exc_info
from inspect import isgeneratorfunction

# XXX the following stream of consciousness is a bit stale.
# What do we want?
# 1. the bot can be in two loop modes _open_ and _closed_.
#  In the open mode the bot executes the prescribed sequence of actions without
#  feedback from the environment. In the closed mode the bot responds to the
#  stimulus and the recent observed state.
# 2. the interaction logic with gym's envs is designed for agents, operating in
#  the closed loop mode.
# 3. Essentially, the bot must accept feedback by default. Hence, the open loop
#  must just be the programmer's choice to ignore the responses form the env 
#  being interacted with.
# Format of the bot's internal responses
# 1. we can use a global serial bus (a queue) onto which the bot's logic pushes
#  new actions, and from which the bot itself pops the actions to be executed.
#  This sounds quite odd... What we really want is a "thread" with loops, that
#  may have their own subloops. The point here is that at each level of nesting
#  the context must have access to the most recent observation to be able to
#  overtake control, abort itself, or continue delegating to the subloop.


def run(gfn, init):
    """a version of `yield from` for the coroutines that can yield other coros.
    """

    # we use an explicit stack here, because we need to globally maintain
    #  the current `obs`, yet be able to switch the generator on-the-fly.
    #  If we were to use recursion, we then would have to somehow propagate
    #  the locally updated `obs` back through all parent frames.

    stack, emergency = [], None
    gen, obs = gfn(init), None
    while True:
        try:
            # purge the stack as we are shutting down
            if emergency is GeneratorExit:
                gen.close()
                if not stack:
                    raise

                gen = stack.pop()
                continue

            # an exception was thrown at us, manually bubble it up
            # through the stack of running generators (gens)
            elif emergency is not None:
                try:
                    gfn = gen.throw(*emergency)

                # the current gen was unable to handle the emergency
                except BaseException:
                    if not stack:
                        raise

                    gen = stack.pop()
                    continue

            else:
                # the current generator either yields an object, or a generator
                # function. In the latter case we should instantiate call it
                # with the most recent `obs`, then suspend ourselves into stack
                # and, finally, delegate control to the new generator..
                gfn = gen.send(obs)

        except StopIteration:
            # the current gen has been exhausted
            if not stack:
                break

            gen = stack.pop()
            continue

        else:
            # we finished the try block and neither encountered any exception
            #  not `continued` prematurely
            emergency = None

        # PEP-342 support (see pytorch PR#49017)
        try:
            # depth-first descend into the sub-generators
            if isgeneratorfunction(gfn):
                stack.append(gen)
                gen, obs = gfn(obs), None

            else:
                obs = yield gfn

        except GeneratorExit:
            emergency = GeneratorExit

        except BaseException:
            emergency = exc_info()

    # end while

    if emergency is not None:
        raise emergency

    # upon termination the raised `StopIteration` communicates
    #  the last `obs` in its `.value`.
    return obs
