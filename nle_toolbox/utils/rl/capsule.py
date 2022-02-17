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


def capsule(learner, n_fragment_length=20, f_h0_lerp=0.05):
    """T-BPTT trajectory collector for capsuled RL. See docs `.engine.step`.

    learner must be a callable object accepting kwargs
        `obs, act, rew, fin, hx`
    returning `act, (val, pol), upd_hx` (diff-able), auto-initing `hx` if
    necessary. Optionally the object could have the `.initial_hx` property for
    the `hx` in the very first call.

    It must also have a method `.learn`, which takes in `obs , act, rew, fin`
    input (always non-diffable), `val, pol` (diff-able depending on the method)
    and `hx` (could be anything, depending on the arch of the learner) and
    updates the internal parameters (possibly including the value in
    `.initial_hx`).
    """
    # (capsule) get the initial recurrent state and `.learn` of the learner
    do_learn = learner.learn
    hx = h0 = getattr(learner, 'initial_hx', None)

    # (capsule) finish handshake and prepare the npyt state
    # XXX not need to create `AliasedNPYT`, since we live in a capsule!
    npy = suply(np.copy, Input(*(yield None)))  # expect obs, act, rew, fin
    pyt = suply(torch.as_tensor, npy)  # XXX `pyt` aliases `npy` (array proto.)
    suply(torch.Tensor.unsqueeze_, pyt, dim=0)  # fake seq dim

    # (sys) perpetual rollout
    fragment = []
    while True:  # .learn()
        # (sys) remember the pre-fragment state `hx` for `.learn` later
        hx_pre = hx

        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        for _ in range(n_fragment_length):
            # (sys) clone for diff-ability, because `pyt` is updated in-place
            input = suply(torch.clone, pyt)

            # REACT x_t, a_{t-1}, h_t -->> a_t, v_t, \pi_t, h_{t+1}
            #       with `a_t \sim \pi_t`
            act_, (val, pol), hx = learner(**input._asdict(), hx=hx)

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

            fragment.append((input, val, pol))

        # (sys) bootstrap the one-step value-to-go approximation, t=N
        # DO NOT yield action to the caller, nor update `npy-pyt`, nor `hx`!
        input = suply(torch.clone, pyt)
        _, (val, pol), _ = learner(**input._asdict(), hx=hx)
        fragment.append((input, val, pol))

        if hx is not None:
            # (sys) retain running state `hx`, but DO NOT backprop through it
            #  form the next fragment (truncated bptt)
            hx = suply(torch.Tensor.detach, hx)

            # (sys) pass some grad feedback from the next fragment into `h0`
            #    `.lerp: hx <<-- (1 - w) * hx + w * h0` (broadcasts correctly).
            if h0 is not None and f_h0_lerp > 0:
                hx = plyr.apply(torch.lerp, hx, h0, weight=f_h0_lerp)

        # (sys) repack data ((x_t, a_{t-1}, r_t, d_t), v_t, \pi_t)
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `pol[t]` is `$\pi_t$
        input, val, pol = plyr.apply(torch.cat, *fragment, _star=False)
        fragment.clear()

        # (sys) learn on the collected fragment with `.learn`
        do_learn(input, val, pol, hx=hx_pre)
