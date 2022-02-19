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


def capsule(step, do, length, *, h0=None, alpha=0.):
    """T-BPTT trajectory collector for capsuled RL. See docs `.engine.step`.

    Parameters
    ----------
    step: callable
        A funciton taking `input` (namedtuple with fields `obs`, `act`, `rew`,
        and `fin`, which represents the recent observation) and the keyword
        `hx` (arbitrarily nested tensors, which store the recurrent runtime
        state, which may be auto-initialised if `hx` is None. `step` returns
        the action `act`, auxiliary data `output`, and the new state `hx`.

    do: callable
        A function that updates whatever internal parameters `step` depends on,
        and takes in `input` (always non-diffable `obs`, `act`, `rew`, `fin`),
        `output` and `hx` (both possibly diff-able).

    length: int
        The length of trajectory fragments used for each truncated bptt grad
        update.

    h0: nested tensors
        The initial learnable runtime state `h0`.

    alpha: float
        The between-fragment blending coefficient, which allows some feedback
        (learning) to `h0` from `hx` used in the collected fragment in T-BPTT.
    """
    if do is None and length >= 1 or do is not None and length < 1:
        raise ValueError('`do` can be None iff fragment `length` is zero.')

    # (sys) let the learner properly init `hx`-s batch dims, since `h0` may
    #  have unitary batch dims
    hx_pre = hx = None

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

        # REACT x_t, a_{t-1}, h_t -->> a_t, y_t, h_{t+1}
        #       with `a_t \sim \pi_t`
        act_, output, hx = step(input, hx=hx)

        # (sys) update the action in `npy` through `pyt`
        suply(torch.Tensor.copy_, pyt.act, act_)

        # STEP \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}
        #      with \omega_t being the unobservable complete state
        obs_, rew_, fin_, nfo_ = yield npy.act

        # (sys) update the rest of the `npy-pyt` aliased context
        # XXX we ignore the info dict `nfo_`, but can report it in fragment!
        suply(np.copyto, npy.obs, obs_)
        suply(np.copyto, npy.rew, rew_)  # XXX allow structured rewards
        np.copyto(npy.fin, fin_)  # XXX must be a boolean scalar/vector

        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        append((input, output))
        if len(fragment) < length:
            continue

        # (sys) one-step lookahead $y_N$, e.g. value-to-go bootstrap.
        # DO NOT yield action to the caller, nor update `npy-pyt`, nor `hx`!
        input = suply(torch.clone, pyt)
        _, output, _ = step(input, hx=hx)
        append((input, output))

        # (sys) repack data ((x_t, a_{t-1}, r_t, d_t), y_t), t=0..N
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `output[t]` is `$y_t$
        input, output = plyr.apply(torch.cat, *fragment, _star=False)
        fragment.clear()

        # (sys) learn on the collected fragment with `.learn`
        do(input, output, hx=hx_pre)

        if hx is not None:
            # (sys) retain running state `hx`, but DO NOT backprop through it
            #  form the next fragment (truncated bptt)
            hx = suply(torch.Tensor.detach, hx)

            # (sys) pass some grad feedback from the next fragment into `h0`
            #    `.lerp: hx <<-- (1 - w) * hx + w * h0` (broadcasts correctly).
            if h0 is not None and alpha > 0:
                hx = plyr.apply(torch.lerp, hx, h0, weight=alpha)

        # (sys) save the recurrent state `hx` at the start of the next fragment
        hx_pre = hx
