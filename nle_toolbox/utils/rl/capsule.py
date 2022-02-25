import plyr
from plyr import suply

import torch
import numpy as np

from .engine import Input


def launch(capsule, initial):
    """Launch the freshly created capsule.
    """
    assert isinstance(initial, Input)

    # (capsule) start with the handshake
    if capsule.send(None) is not None:
        raise RuntimeError('Capsule handshake failed.')

    # (capsule) communicate the initial input and gets the reaction to it
    return capsule.send(initial)


def capsule(step, update, length):
    """T-BPTT trajectory collector for capsuled RL. See docs `.engine.step`.

    Parameters
    ----------
    step: callable
        A funciton taking `input` (namedtuple with fields `obs`, `act`, `rew`,
        and `fin`, which represents the recent observations) and the keyword
        `hx` (arbitrarily nested tensors), which stores the recurrent runtime
        state, that may be auto-initialised if `hx` is None. `step` returns
        the action `act`, auxiliary data `output`, and the new state `hx`.

    update: callable
        A function that updates whatever internal parameters `step` depends on,
        and takes in `input` (always non-diffable `obs`, `act`, `rew`, `fin`),
        `output`, `hx` and `gx` (possibly diff-able). It returns an auxiliary
        information dict and an update for `hx` (may be diff-able).

    length: int
        The length of trajectory fragments used for each truncated bptt grad
        update.
    """
    if update is None and length >= 1 or update is not None and length < 1:
        raise ValueError('`update` can be None iff fragment `length` is zero.')

    # (sys) let the learner properly init `hx`-s batch dims
    # XXX hx is current, gx is at the start of the fragment
    gx = hx = None

    # (capsule) finish handshake and prepare the npyt state
    # XXX not need to create `AliasedNPYT`, since we live in a capsule!
    npy = suply(np.copy, Input(*(yield None)))  # expect obs, act, rew, fin
    pyt = suply(torch.as_tensor, npy)  # XXX `pyt` aliases `npy` (array proto.)
    suply(torch.Tensor.unsqueeze_, pyt, dim=0)  # fake seq dim

    # (sys) collect trajectory in fragments, when instructed to
    fragment = []
    append = id if length < 1 else fragment.append  # `id` serves as a dummy

    # (sys) perpetual rollout
    while True:  # .learn()
        # (sys) clone for diff-ability, because `pyt` is updated in-place
        input = suply(torch.clone, pyt)

        # REACT x_t, a_{t-1}, h_t -->> a_t, y_t, h_{t+1} with `a_t \sim \pi_t`
        #  XXX if the runtime state is irrelevant to `step`, then it returns
        #  `hx` intact
        act_, output, hx = step(input, hx=hx)

        # (sys) update the action in `npy` through `pyt`
        suply(torch.Tensor.copy_, pyt.act, act_)

        # STEP \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}
        #      with \omega_t being the unobservable complete state
        obs_, rew_, fin_, nfo_ = yield npy.act
        # XXX adding a skip logic here is dumb: just don't .send anything into
        # this capsule!

        # (sys) update the rest of the `npy-pyt` aliased context
        # XXX we ignore the info dict `nfo_`, but can report it in fragment!
        suply(np.copyto, npy.obs, obs_)
        suply(np.copyto, npy.rew, rew_)  # XXX allow structured rewards
        np.copyto(npy.fin, fin_)  # XXX must be a boolean scalar/vector

        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        append((input, output))
        if len(fragment) < length:
            continue

        # (sys) one-step look-ahead $y_N$, e.g. value-to-go bootstrap.
        # DO NOT yield action to the caller, nor update `npy-pyt`, nor `hx`!
        input = suply(torch.clone, pyt)
        _, output, _ = step(input, hx=hx)
        append((input, output))

        # (sys) repack data ((x_t, a_{t-1}, r_t, d_t), y_t), t=0..N
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `output[t]` is `$y_t$
        input, output = plyr.apply(torch.cat, *fragment, _star=False)
        fragment.clear()

        # (sys) do an update on the collected fragment and get the revised
        # recurrent runtime state for the next fragment
        _, gx = update(input, output, gx=gx, hx=hx)
        hx = gx = hx if gx is None else gx  # if None, set gx to hx
