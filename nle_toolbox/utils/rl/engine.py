import plyr

import torch
import numpy as np

from torch.nn import functional as F

from collections import namedtuple

import gym
from gym.vector.utils import batch_space


class SerialVecEnv(gym.Env):
    """A simple serial vectorized env."""

    def __init__(self, factory, n_envs=1):
        self.envs = [factory() for _ in range(n_envs)]

        # assume the facory produces consistent envs
        env = self.envs[0]
        self.observation_space = batch_space(env.observation_space, n_envs)
        self.action_space = batch_space(env.action_space, n_envs)

    def seed(self, seed):
        seeds = []
        ss = np.random.SeedSequence(seed)
        for env, seed in zip(self.envs, ss.spawn(len(self.envs))):
            seeds.append(env.seed(seed))

        # return the same seed, since it can be used to re-seed the envs
        return seed, seeds

    def reset(self, **kwargs):
        obs = (env.reset(**kwargs) for env in self.envs)
        return plyr.apply(np.stack, *obs, _star=False, axis=0)

    def close(self):
        for env in self.envs:
            env.close()

    def step(self, actions):
        result = []
        for j, env in enumerate(self.envs):
            act = plyr.apply(plyr.getitem, actions, index=j)
            obs, rew, fin, nfo = env.step(act)
            if fin:
                obs = env.reset()

            # `obs` is s_0 if `fin` else s_{t+1}, `rew` is r_{t+1}
            result.append((obs, rew, fin, nfo))

        obs_, rew_, fin_, nfo = zip(*result)
        obs = plyr.apply(np.stack, *obs_, _star=False, axis=0)
        return obs, np.array(rew_), np.array(fin_), nfo

    @property
    def nenvs(self):
        return len(self.envs)


# stripped down version or rlplay.engine
Input = namedtuple('Input', 'obs,act,rew,fin')
# XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
#  e.g. `.rew[t]` is $r_t$, and `pol[t]` is `$\pi_t$.

# A pair of host-resident numpy array and torch tensor with aliased data
AliasedNPYT = namedtuple('AliasedNPYT', 'npy,pyt')
# XXX converting array data between numpy and torch is zero-copy (on host)
#  due to the __array__ protocol, i.e. both can alias each other's storage,
#  meaning that updates of one are immediately reflected in the other.


def prepare(env, rew=np.nan, fin=True):
    # we rely on an underlying Vectorized Env to supply us with correct
    #  product action space, observations, and a number of environments.
    assert isinstance(env, SerialVecEnv)

    # prepare the runtime context (coupled numpy-torch tensors)
    npy = Input(
        env.reset(),
        env.action_space.sample(),
        np.full(env.nenvs, rew, dtype=np.float32),
        np.full(env.nenvs, fin, dtype=bool),
    )

    # in-place unsequeeze produces a writable view, which preserves aliasing
    pyt = plyr.apply(torch.as_tensor, npy)
    plyr.apply(torch.Tensor.unsqueeze_, pyt, dim=0)

    return AliasedNPYT(npy, pyt)


def step(env, agent, npyt, hx):
    r"""Perform the `t` -->> `t+1` env's transition under the agnet's policy.

    Details
    -------
    Assuming the true unobserved state of the ENV is `\omega_t`,
    the aliased `npy`-`pyt` contains the recent observation `x_t`,
    action `a_{t-1}`, reward `r_t` and termination flag `d_t`,
    and `hx` is `h_t` the current recurrent state of the agent,
    this procedure does the agent's REACT step
    $$
        (x_t, a_{t-1}, r_t, d_t), h_t -->>  v_t, \pi_t, h_{t+1}
        \,, $$

    then samples the action `a_t \sim \pi_t` and follows by the ENV step
    $$
        \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r^E_{t+1}, d_{t+1}
        \,. $$

    Returns the t-th step afterstate
    $$
        ((x_t, a_{t-1}, r_t, d_t), v_t, \pi_t), h_{t+1}
        \,, $$

    and updates the aliased npy-pyt INPLACE to contain
    $$
        (x_{t+1}, a_t, r_{t+1}, d_{t+1})
        \,. $$

    Note
    ----
    This is not supposed to be used in multiprocessing.
    """
    npy, pyt = npyt

    # (sys) clone to avoid graph diff-ability issues, because
    #  `pyt` is updated IN-PLACE through storage-aliased `npy`.
    input = plyr.apply(torch.clone, pyt)

    # (agent) REACT x_t, a_{t-1}, h_t -->> v_t, \pi_t, h_{t+1}
    #  and sample `a_t \sim \pi_t`
    act_, (val, pol), hx = agent(**input._asdict(), hx=hx)

    # (sys) update the action in `npy` through `pyt`
    plyr.apply(torch.Tensor.copy_, pyt.act, act_)

    # (env) STEP \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r^E_{t+1}, d_{t+1}
    # XXX we assume the env performs auto-resetting steps
    obs_, rew_, fin_, nfo_ = env.step(npy.act)

    # (sys) update the rest of the npy/pyt running state
    plyr.apply(np.copyto, npy.obs, obs_)
    np.copyto(npy.rew, rew_)
    np.copyto(npy.fin, fin_)
    # XXX we ignore the info dict `nfo_`

    return (input, val, pol), hx


def pyt_polgrad(logpol, act, adv):
    r"""Compute the GAE policy gradient surrogate.

    Details
    -------
    The policy `logpol` is a `(T + 1) x B x ...` float tensor with NORMALIZED
    `\log \pi_t` logits, while the actions `act` is a long tensor `(T + 1) x B`
    containing the indices of the taken actions `a_{t-1}` (one step lag
    behind `logpol`). The scores `A_t` of the policy grad surrogate are in
    `adv` -- a `T x B x ...` float tensor. `A_t` is computed from the future
    rewards and value-to-go estimates `(v_j, r_{j+1})_{j \geq t}`.
    """
    # the policy contains logits over the last (event) dims
    # \sum_j \sum_t A_{j t} \log \pi_{j t}(a_{j t})
    logp = logpol[:-1].gather(-1, act[1:].unsqueeze(-1))
    return logp.squeeze(-1).mul(adv).sum()


def pyt_entropy(logpol):
    r"""Compute the entropy `- \sum_j \pi_{j t} \log \pi_{j t}` over
    the event dim.
    """
    # XXX `.new_zeros(())` creates a scalar zero (yes, an EMPTY tuple)
    entropy = F.kl_div(
        logpol.new_zeros(()),
        logpol[:-1],
        log_target=True,
        reduction='none',
    ).sum(dim=-1).neg()

    # sum over the remaining dims
    return entropy.sum()


def pyt_critic(val, ret):
    r"""The critic loss in the A2C algo (and others)."""
    mse = F.mse_loss(
        val[:-1],
        ret,
        reduction='none',
    ).sum(dim=-1)

    return mse.sum()
