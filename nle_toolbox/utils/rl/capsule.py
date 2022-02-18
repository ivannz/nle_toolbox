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


def capsule(learner, n_fragment_length=20, f_h0_lerp=0.):
    """T-BPTT trajectory collector for capsuled RL. See docs `.engine.step`.

    learner must be a callable object accepting kwargs
        `obs, act, rew, fin, hx`,
    returning `act, output, upd_hx` (diff-able), auto-initialising `hx` if
    necessary. Optionally the object could have the `.initial_hx` property
    for the `hx` in the very first call. Here `hx` is the recurrent (runtime)
    state of the learner.

    It must also have a method `.learn`, which takes in `obs , act, rew, fin`
    input (always non-diffable), `output` (diff-able depending on the method)
    and `hx` (could be anything, depending on the arch of the learner) and
    updates the internal parameters (possibly including the value in
    `.initial_hx`).
    """
    # (capsule) get the initial recurrent (runtime) state and `.learn` of the learner
    do_learn = learner.learn
    h0 = getattr(learner, 'initial_hx', None)

    # (sys) let the learner properly init `hx`-s batch dims, since `h0` may
    #  have unitary batch dim
    hx_pre = hx = None

    # (capsule) finish handshake and prepare the npyt state
    # XXX not need to create `AliasedNPYT`, since we live in a capsule!
    npy = suply(np.copy, Input(*(yield None)))  # expect obs, act, rew, fin
    pyt = suply(torch.as_tensor, npy)  # XXX `pyt` aliases `npy` (array proto.)
    suply(torch.Tensor.unsqueeze_, pyt, dim=0)  # fake seq dim

    # (sys) collect trajectory in fragments, when instructed to
    fragment = []
    append = (lambda x: None) if n_fragment_length < 1 else fragment.append

    # (sys) fixup length to make the fragment size check work properly
    n_fragment_length = max(1, n_fragment_length)

    # (sys) perpetual rollout
    while True:  # .learn()
        # (sys) clone for diff-ability, because `pyt` is updated in-place
        input = suply(torch.clone, pyt)

        # REACT x_t, a_{t-1}, h_t -->> a_t, y_t, h_{t+1}
        #       with `a_t \sim \pi_t`
        act_, out, hx = learner(**input._asdict(), hx=hx)

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

        append((input, out))

        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        if len(fragment) < n_fragment_length:
            continue

        # (sys) one-step lookahead $y_N$, e.g. value-to-go bootstrap.
        # DO NOT yield action to the caller, nor update `npy-pyt`, nor `hx`!
        input = suply(torch.clone, pyt)
        _, out, _ = learner(**input._asdict(), hx=hx)
        fragment.append((input, out))

        # (sys) repack data ((x_t, a_{t-1}, r_t, d_t), y_t), t=0..N
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `out[t]` is `$y_t$
        input, out = plyr.apply(torch.cat, *fragment, _star=False)
        fragment.clear()

        # (sys) learn on the collected fragment with `.learn`
        do_learn(input, out, hx=hx_pre)

        if hx is not None:
            # (sys) retain running state `hx`, but DO NOT backprop through it
            #  form the next fragment (truncated bptt)
            hx = suply(torch.Tensor.detach, hx)

            # (sys) pass some grad feedback from the next fragment into `h0`
            #    `.lerp: hx <<-- (1 - w) * hx + w * h0` (broadcasts correctly).
            if h0 is not None and f_h0_lerp > 0:
                hx = plyr.apply(torch.lerp, hx, h0, weight=f_h0_lerp)

        # (sys) save the recurrent state `hx` at the start of the next fragment
        hx_pre = hx
