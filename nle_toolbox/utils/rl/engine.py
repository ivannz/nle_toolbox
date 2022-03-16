import plyr

import torch
import numpy as np

from torch.nn import functional as F

from copy import deepcopy
from collections import namedtuple

import gym
from gym.vector.utils import batch_space


class SerialVecEnv(gym.Env):
    """A simple serial vectorized env.
    """
    def __init__(
        self,
        factory: callable,
        n_envs: int = 1,
        *,
        args: tuple = (),
        kwargs: dict = None,
    ):
        self.envs = [factory(*args, **(kwargs or {})) for _ in range(n_envs)]

        # assume the factory produces consistent envs
        env = self.envs[0]
        self.observation_space = batch_space(env.observation_space, n_envs)
        self.action_space = batch_space(env.action_space, n_envs)

    def seed(self, seed):
        # XXX does not work for envs, which use non-numpy or legacy PRNG
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
            # `act` is a_t
            act = plyr.apply(plyr.getitem, actions, index=j)

            # `obs` is x_{t+1}, `rew` is r_{t+1}, `fin` is True if terminal
            # `nfo` is auxiliary dict, related to `t -->> t+1` transition
            obs, rew, fin, nfo = env.step(act)
            if fin:
                obs = env.reset()

            # `obs` is x_0 if `fin` else x_{t+1}
            result.append((obs, rew, fin, nfo))

        obs_, rew_, fin_, nfo_ = zip(*result)
        # MAYBE the info dict `nfo` should NOT be repacked (unlike other data),
        #  since it may contain dynamic auxiliary diagnostic information it is
        #  up to env's implementation.
        nfo = plyr.apply(plyr.AtomicTuple, *nfo_, _star=False)
        obs = plyr.apply(np.stack, *obs_, _star=False, axis=0)
        return obs, np.array(rew_), np.array(fin_), nfo

    @property
    def nenvs(self):
        return len(self.envs)

    def __len__(self):
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
    npy = plyr.apply(np.copy, Input(
        env.reset(),
        env.action_space.sample(),
        # pre-filled arrays for potentially structured rewards
        plyr.ragged(np.full, len(env), rew, dtype=np.float32),
        # `fin` is an array and NEVER structured
        np.full(len(env), fin, dtype=bool),
    ))

    # in-place unsequeeze produces a writable view, which preserves aliasing
    pyt = plyr.apply(torch.as_tensor, npy)
    plyr.apply(torch.Tensor.unsqueeze_, pyt, dim=0)

    return AliasedNPYT(npy, pyt)


def step(env, agent, npyt, hx, *, device=None):
    r"""Perform the `t` -->> `t+1` env's transition under the agnet's policy.

    Details
    -------
    Assume the env is currently at the _true unobserved state_ $\omega_t$ and
    the `npyt` aliased context contains the recent $(x_t, a_{t-1}, r_t, d_t)$.
    Letting `hx` be $h_t$ the recurrent, or _runtime_ state of the agent, this
    procedure, first, does the agent's __REACT__ step
    $$
        (x_t, a_{t-1}, r_t, d_t), h_t -->>  a_t, y_t, h_{t+1}
        \,, $$

    where $a_t \sim \pi_t$, $y_t$ is its output (agent's state-value estimate
    $v_t$ and policy $\pi_t$), and $h_{t+1}$ is its new runtime state, and then
    performs the env's __ENV__ step
    $$
        \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}, I_{t+1}
        \,, $$

    where $\omega_{t+1}$ is the updated _true unobserved_ state of the env,
    $d_{t+1}$ is the termination flag, $r_{t+1}$ is the reward due to the just
    made transition, and $x_{t+1}$ is the newly emitted observation. $I_{t+1}$
    is the info dict, which contains auxiliary data associated with the step.

    Returns the $t$-th step afterstate
    $$
        ((x_t, a_{t-1}, r_t, d_t), y_t), h_{t+1}
        \,, $$

    updates the aliased `npyt` INPLACE to contain
    $$
        (x_{t+1}, a_t, r_{t+1}, d_{t+1})
        \,, $$

    and causes a side effect UPDATE in `env` by calling its `.step`.

    Important
    ---------
    This procedure DEFINES the data synchronisation in the key and aux arrays,
    which is ASSUMED and RELIED ON by all functions in this submodule, e.g.
    by `pyt_polgrad()`.

    Key arrays correspond to historical trajectory states (past and present):
      * `obs` $x_t$ is the _present_ observation (at time $t$);
      * `act` $a_{t-1}$ is the action, that _led_ to this observation;
      * `rew` $r_t$ is the reward emitted _alongside_ $x_t$ for $a_{t-1}$;
      * `fin` $d_t$ is indicates if the action terminated the episode;

    Aux arrays contain data computed GIVEN the past and present historical
    data, but BEFORE the future:
      * `out` the state-value estimate $v_t$ and policy $\pi_t$
      * `hx` $h_{t+1}$ is the agent's runtime state

    The only data that is in the FUTURE relative to everything else
    is the info dict `nfo_` $I_{t+1}$, returned by the `.step`.

    Note
    ----
    This is not supposed to be used in multiprocessing.
    """
    npy, pyt = npyt

    # (sys) clone to avoid graph diff-ability issues, because `pyt` is updated
    #  INPLACE through storage-aliased `npy`
    if device is None or device.type == 'cpu':
        input = plyr.apply(torch.clone, pyt)

    else:
        # host-device moves produce a copy, since `pyt` is on the HOST!
        input = plyr.apply(lambda t: t.to(device), pyt)

    # (agent) REACT x_t, a_{t-1}, h_t -->> v_t, \pi_t, h_{t+1}
    #  and sample `a_t \sim \pi_t`
    act_, out, hx = agent(**input._asdict(), hx=hx)

    # (sys) update the action in `npy` through `pyt`
    plyr.apply(torch.Tensor.copy_, pyt.act, act_)

    # (env) STEP \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r^E_{t+1}, d_{t+1}
    # XXX we assume the env performs auto-resetting steps (see `SerialVecEnv`)
    obs_, rew_, fin_, nfo_ = env.step(npy.act)

    # (sys) update the rest of the `npy-pyt` aliased context
    #      `fin` is an array and NEVER structured.
    plyr.apply(np.copyto, npy.obs, obs_)
    plyr.apply(np.copyto, npy.rew, rew_)  # XXX allow structured rewards
    np.copyto(npy.fin, fin_)  # XXX must be a boolean scalar/vector

    # Make adeep copy of the info dict `nfo_` before returning it
    return (input, out), hx, deepcopy(nfo_)  # ATTN `nfo` is AFTER `input`


def dropout_mask(input, *, k=None):
    # (batching) instead of sampling a random batch from the specified buffer
    #  we do the next best thing and sample a binary mask which ignores certain
    #  loss terms. For a non-recurrent agent this is entirely equivalent to
    #  sampling a random batch w/o replacement (according to the mask), except
    #  for the full O(TB) arithmetic complexity. For recurrent agents this
    #  allows proper feedback propagation for the recurrent states, due to
    #  multiple grad pathways for $h_t$, $t=0..T$, in $
    #      y_t, h_{t+1} = g(x_t, h_t; \theta)
    #  $.

    fin = input.fin[1:]
    if isinstance(k, float) and 0 < k < 1:
        # k is a float within the zero-one interval means we want
        # "batch dropout": a mask with `1` w. prob `p`.
        uni = torch.rand(fin.shape, device=fin.device)
        return uni.le(k)

    elif isinstance(k, int) and 0 < k < fin.numel():
        # draw a sample of uniform rv-s, then set to zero
        #  all values that are not among the `k` least ones.
        uni = torch.rand(fin.shape, device=fin.device)
        return uni.le(uni.flatten().kthvalue(k).values)

    elif k is None:
        return None

    raise ValueError("`k` must be either `None`, a +ve int,"
                     f" or a float between 0 and 1. Got `{k}`.")


def pyt_logpact(logpol, act):
    """Get log probability of the taken actions.

    Details
    -------
    This functions relies on its arguments having the correct synchronisation
    between `logpol` and `act`
    """
    # (sys) get \log\mu_t(a_t) from `logpol[t][act[t+1]])`, t=0..T-1
    return logpol[:-1].gather(-1, act[1:].unsqueeze(-1)).squeeze_(-1)


def pyt_polgrad(logpol, act, adv, *, mask=None):
    r"""Compute the GAE policy gradient surrogate.

    Details
    -------
    This functions relies on its arguments having the correct synchronisation.

    The policy `logpol` is a `(T + 1) x B x ...` float tensor with NORMALIZED
    `\log \pi_t` logits, while the actions `act` is a long tensor `(T + 1) x B`
    containing the indices of the taken actions `a_{t-1}` (one step lag
    behind `logpol`). The scores `A_t` of the policy grad surrogate are in
    `adv` -- a `T x B x ...` float tensor. `A_t` is computed from the future
    rewards and value-to-go estimates `(v_j, r_{j+1})_{j \geq t}`.
    """
    # the policy contains logits over the last (event) dims
    # \sum_j \sum_t A_{j t} \log \pi_{j t}(a_{j t})
    out = pyt_logpact(logpol, act).mul(adv)

    # block the grads trough certain surrogate losses
    if mask is not None:
        scale = float(mask.sum()) / mask.numel()
        out = out.mul(mask).div(scale)

    return out.sum()


def pyt_entropy(logpol, *, mask=None):
    r"""Compute the entropy `- \sum_j \pi_{j t} \log \pi_{j t}` over
    the event dim.
    """
    # `.kl_div` correctly handles `-ve` infinite logits (or zero probs), but
    #  computes $\sum_j e^{\log p_j} \log p_j$, so we need to flip the sign.
    # XXX `.new_zeros(())` creates a scalar zero (yes, an EMPTY tuple)
    # XXX `.kl_div` computes \sum_n e^y_n (y_n - x_n), for y = logpol, x = 0.
    entropy = F.kl_div(
        logpol.new_zeros(()),
        logpol[:-1],
        log_target=True,
        reduction='none',
    ).sum(dim=-1).neg()

    # sum over the remaining dims after applying the optional mask
    if mask is not None:
        scale = float(mask.sum()) / mask.numel()
        entropy = entropy.mul(mask).div(scale)

    return entropy.sum()


def pyt_critic(val, ret, *, mask=None):
    r"""The critic loss in the A2C algo (and others).
    """
    mse = F.mse_loss(
        val[:-1],
        ret,
        reduction='none',
    )

    if mask is not None:
        scale = float(mask.sum()) / mask.numel()
        mse = mse.mul(mask).div(scale)

    return mse.sum() / 2
