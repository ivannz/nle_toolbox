import plyr
from plyr import suply

import torch
import numpy as np

from .engine import Input, AliasedNPYT


def collect(fragment, agent, npyt, hx):
    """Trajectory collector for capsuled RL. See docs `.engine.step`.
    """
    npy, pyt = npyt
    for n in range(len(fragment)):
        # (sys) clone for diff-ability, because `pyt` is updated in-place
        input = suply(torch.clone, pyt)

        # (agent) REACT x_t, a_{t-1}, h_t -->> v_t, \pi_t, h_{t+1}
        #  and sample `a_t \sim \pi_t`
        act_, (val, pol), hx = agent(**input._asdict(), hx=hx)

        # (sys) update the action in `npy` through `pyt`
        suply(torch.Tensor.copy_, pyt.act, act_)

        # (env) STEP \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}
        obs_, rew_, fin_, nfo_ = yield npy.act

        # (sys) update the rest of the `npy-pyt` aliased context
        # XXX we ignore the info dict `nfo_`, but can report it in fragment!
        suply(np.copyto, npy.obs, obs_)
        np.copyto(npy.rew, rew_)
        np.copyto(npy.fin, fin_)

        fragment[n] = (input, val, pol), hx


def capsule(learner, n_fragment_length=20, f_h0_lerp=0.05):
    # Finish handshake and prepare the npyt state
    npy = Input(*(yield None))  # expect obs, act, rew, fin
    npyt = AliasedNPYT(npy, suply(torch.as_tensor, npy))
    suply(torch.Tensor.unsqueeze_, npyt.pyt, dim=0)  # fake seq dim

    h0, hx = learner.initial_hx, None
    while True:  # .learn()
        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        # XXX `collect()` updates `buffer` and `npyt` in-place
        buffer = [None] * n_fragment_length
        yield from collect(buffer, learner, npyt, hx)

        # (sys) separate inputs from the recurrent states
        fragment, hxx = zip(*buffer)
        del buffer

        # (sys) bootstrap the one-step value-to-go approximation
        # DO NOT step through env, nor update `npyt`, nor `hx`!
        input = suply(torch.clone, npyt.pyt)
        _, (val, pol), _ = learner(**input._asdict(), hx=hxx[-1])
        fragment += ((input, val, pol),)

        # (sys) repack data ((x_t, a_{t-1}, r^E_t, d_t), v_t, \pi_t)
        input, val, pol = plyr.apply(torch.cat, *fragment, _star=False)
        del fragment
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `pol[t]` is `$\pi_t$.

        # (sys) retain running state `hx`, but DO NOT backprop through it
        #  form the next fragment (truncated bptt)
        hx_upd = suply(torch.Tensor.detach, hxx[-1])
        del hxx

        # (sys) learn on the collected fragment and the finished episodes. Also
        #  let the learner mess with the non-diffable `hx` for the next fragment.
        learner.learn(input, val, pol, hx=hx)

        # (sys) pass some grad feedback from the next fragment into `h0`
        # by lerp-ing from `hx` to `h0`.
        #    `.lerp: hx <<-- (1 - w) * hx + w * h0` (broadcasts correctly).
        if h0 is not None and f_h0_lerp > 0:
            hx_upd = plyr.apply(torch.lerp, hx_upd, h0, weight=f_h0_lerp)
        hx = hx_upd


def launch(capsule, obs, act, n_envs):
    # start with the handshake
    if capsule.send(None) is not None:
        raise RuntimeError('Capsule handshake failed.')

    # init the capsule: communicate the env specs and gets the first reaction.
    rew = np.full(n_envs, 0., dtype=np.float32)
    fin = np.full(n_envs, True, dtype=bool)
    return capsule.send((obs, act, rew, fin))
