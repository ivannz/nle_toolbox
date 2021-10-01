# Abstraction Layer design

Rules:
* no event loop is allowed to steal execution control
* each layer may react be either processing the event itself, or passing it to
its sub-layers downstream
* the _generator functions_ (`gf`) have _the latest received input_ (`rx`) as
their first positional argument. This ensures that when immediately started
with `.send(None)` the _generator-iterator_ (`gi`) yields a value that is a proper
in _response_. Afterwards, `gi.send(rx)` operate normally: the suspended `gi` is
resumed at `yield` with a new received input `rx`, to which it reacts either with
a new value at the next `yield`, an exhaustion event upon hitting a `return ...`
(`StopIteration`), or with a raised exception.
* the loop itself knows when it can or cannot be run, it is is responsibility!
* the dispatcher implements the selection strategy (maybe the sub-loops can bid on
a state, but there is a possibilty of value misrepresentaition... second price
auction perhaps?)


## Event loop

Event loop with a functon handler and continuation condition is the simplest
most basic mechanism. Its logic underlies everything.
```python
# `react` returns an object
def loop(rx, *, check, react):
    # while the rx is valid, process it and yield a result
    while check(rx):
        rx = yield react(rx)
        # XXX `react` can dispatch `rx` to a proper function handler and
        #  return its response.
```

## Adaptor Wrapper

Adaptor cannot be reduced to an event loop with a `react` that _does not_
resemble a nested loop like the one below.
```python
def adaptor(wrapped, rx, *, check, fwd, rev):
    # start the wrapped loop. While valid, `fwd` transform `rx`, feed it
    #  into the loop, and yield a `rev` transformed returned value.
    try:
        gi, tx = wrapped(fwd(rx)), None
        while check(rx):
            rx = yield rev(gi.send(tx))
            tx = fwd(rx)

    except StopIteration as e:
        raise RuntimeError(e.value) from None
```

Indeed, if we were to express `adaptor` through a `loop` the tentative `react` would
have to be stateful, i.e. being able to remember and access a suspended generator `wrapped`.
```python
rev, fwd, wrapped, check = ...


def _react(rx, *, self={}):  # noqa: persistent state through a "singleton" dict
    try:
        if 'gi' not in self:
            gi = self['gi'] = wrapped(fwd(rx))
            return rev(gi.send(None))

        return rev(self['gi'].send(fwd(rx)))

    except StopIteration as e:
        raise RuntimeError(e.value) from None


def adaptor(rx):
    yield from loop(rx, check=check, react=_react)
```

## Dispatcher

A dispatcher's `react` returns a generator factory depending on the current
`rx`. Like the adaptor above, it is unreasonable to represent is as an event
loop, since its react would have to look like the `dispatcher` code below:
```python
def is_suspended(gen):
    if gen.gi_frame is None:
        return False

    # check the index of the last instruction in the generator's frame.
    #  see `inspect.getgeneratorstate`
    return gen.gi_frame.f_lasti != -1


def dispatcher(rx, *, react):
    # dispatcher reacts with either a generator function (a generator creator).
    # and it is our responsibility to either identify an existing suspended
    # generator from it, or get a properly initted generator and start it.
    try:
        active, gf = {}, react(rx)
        while gf is not None:
            if gf not in active:  # spawned generators live in the local state
                active[gf] = gf(rx)

            gi = active[gf]
            rx = yield gi.send(rx if is_suspended(gi) else None)
            # XXX atm we raise RuntimeError if a sub-generator stops iterating,
            # but is there a better solution?

            gf = react(rx)

    except StopIteration as e:
        raise RuntimeError(e.value) from None
```
Dispatcher's `react` must implement a selection strategy over the generator feactories
based on the supplied `rx`.

Still, how would a dispatcher's implementation through `loop` look like?
```python
check, react = ...


def _check(rx):
    return react(rx) is not None


def _react(rx):
    try:
        gi = react(rx)
        return gi.send(rx if is_suspended(gi) else None)

    except StopIteration as e:
        raise RuntimeError(e.value) from None


def dispatcher(rx):
    yield from loop(rx, check=_check, react=_react)
```


## Filter

The layered/shelled structure naturally lends itslef to the conecpt of event filtering.
A filter is essentially a binary dispatcher with one loop fused into itself. Also it
can be thought of as an adaptor with a mind `react` of its own.

```python
def filter(rx, *, check, react, wrapped):
    try:
        gi = None
        while True:
            if check(rx):
                tx = react(rx)

            else:
                if gi is None:
                    # init `gi` when it is its first time to react
                    gi, rx = wrapped(rx), None  # reset `rx` to fire up gi
                tx = gi.send(rx)

            rx = yield tx

    except StopIteration as e:
        raise RuntimeError(e.value) from None
```


## An event handler loop based on a nerual module

So we do not have to burden each `react` with a logic that creates instances of complex objects,
inits their and starts thrir generators.

Below is some code for a neural control of some gym environment. We set up a filter and
an inner neural control policy.
```python
import gym
import torch


class NeuralShell(Shell):
    def __init__(self, ...):
        self.core, self.features, self.terminator, self.control = ...

    @torch.no_grad()  # see the reminder below
    def loop(self, rx, /, *):
        # a generator function that yields `tx` in response to the latest `rx`.

        x, hx = self.core(self.features(rx), hx=None)  # recurrent core
        while bool(self.terminator(x)):
            rx = yield self.control(x)

            x, hx = self.core(self.features(rx), hx)


ns = NeuralShell(...)
check, react = ...

with gym.make(...) as env:
    # prepare and init
    obs, fin, rew = env.reset(), False, 0.
    gi = filter(obs, check=check, react=react, wrapped=ns.loop)

    # fire up and loop (see `shell-proto.py`)
    obs = None
    while bool(env.render('human')) and not fin:  # bool expr order matters!
        obs, rew, fin, info = env.step(gi.send(obs))
```

# Trunk

## Thoughts

Vickrey-Clarke-Groves auction for strategy selection
  * we need an adversarial approach here: the dispatcher and the dispatched compete.
  * the dispatched _maximize_ their own reward, which they can get only if they are
  being executed
  * thus each one places a bid for the execution at the current state, however there also
  needs to be a true value function of the dispatched...
  * the dispatcher must try to reveal the true values of the dispatched

  check lit...

This would sound novel, but the problem here is the design: under VCG bidding $v_i(s_t)$,
i.e. true the state value function, is optimal, hence there is no need to even consider
a separate bidding srat $b_i(s_t)$. However, $v_i$ is the evaluation of $s_t$, i.e
launching a traj $s_{k\mid t}$ from $s_{t\mid t} = s_t$ following the $i$-th sub-policy.
But this is exactly the definition of the q-factor $q(s_t, i)$ for the dispatcher, who
picks the $\arg\max_i q(s_t, i)$, just like the winnig bid in VCG.

## A reminder

```python
import torch


def foo():
    with torch.no_grad():
        print('foo', torch.is_grad_enabled())
        yield 1
    print('foo', torch.is_grad_enabled())
    yield 2


@torch.no_grad()
def baz():
    print('baz', torch.is_grad_enabled())
    yield 1
    print('baz', torch.is_grad_enabled())
    yield 2


with torch.enable_grad():
    gi = foo()
    print(torch.is_grad_enabled())
    next(gi)
    print(torch.is_grad_enabled())
    next(gi)
    print(torch.is_grad_enabled())


with torch.enable_grad():
    gi = baz()
    print(torch.is_grad_enabled())
    next(gi)
    print(torch.is_grad_enabled())
    next(gi)
    print(torch.is_grad_enabled())
```
