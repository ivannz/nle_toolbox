import plyr
import torch
import numpy as np

import operator as op
from functools import wraps, partial
from collections import namedtuple
from itertools import chain

from time import monotonic_ns

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from nle_toolbox.utils.nn import multinomial
from nle_toolbox.utils.nn import ModuleDict, ModuleDictSplitter
from nle_toolbox.utils.nn import masked_rnn, InputGradScaler

import gym
import nle_toolbox.utils.rl.engine as rl
from nle_toolbox.utils.rl.capsule import buffered, launch
from nle_toolbox.utils.rl.returns import pyt_ret_gae, gamma
from nle_toolbox.utils.rl.alg import extract_truncated

from nle_toolbox.utils.rl.tools import EpisodeExtractor
from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter

import pytest


pytestmark = pytest.mark.skip(reason="currently this is not a test suite")


def linear(t, t0=0.0, t1=1.0, v0=0.0, v1=1.0):
    tau = min(max(t, t0), t1)
    return ((tau - t0) * v1 + (t1 - tau) * v0) / (t1 - t0)


def const(t, v=1.0):
    return v


@dataclass
class Config:
    # should we make a Config -->> Schedule transform?
    n_total_steps: int = 3_000_000
    n_fragment_length: int = 1024
    n_envs: int = 16
    b_ternary: bool = True

    f_model_lerp: float = 1.0

    n_act_embedding_dim: int = 0
    b_recurrent: bool = False
    f_lr: float = 3e-4
    f_optim_eps: float = 1e-5  # detail #?: it turns out, this is very important!

    f_gam: float = 0.999
    n_ppo_batch_size: int = 64
    n_seq: int = 1  # 10
    n_ppo_epochs: int = 4  # 2

    f_ppo_eps: float = 0.2
    b_adv_normalization: bool = True
    f_clip_grad: float = 0.5
    f_entropy: float = 0.01
    f_critic: float = 0.5

    @property
    def f_loss_coef(self) -> dict[str, float]:
        # -ve -->> max, +ve -->> min
        return {
            "polgrad": -1.0,
            "entropy": -self.f_entropy,
            "critic": self.f_critic,
        }

    c_lam_anneal = partial(const, v=0.98)  # f_lam
    # c_lam_anneal = partial(linear, t0=0.05, t1=0.95, v0=0.98, v1=0.5)


config = Config()

ValPolPair = namedtuple("ValPolPair", "val,pol")
Buffer = namedtuple("Buffer", "input,out,nex,adv,ret")


def reduce(values, weight=None):
    flat = []
    if weight is not None:
        values = plyr.apply(op.mul, values, weight)

    plyr.apply(flat.append, values)
    return sum(flat)


def process_timings(timing, *labels, scale=1.0, aggregate=False):
    # collect the timings
    timing = np.reshape(timing, (-1, 1 + len(labels)), "C")

    # get the fraction spent at each step
    timing_label = np.diff(timing, axis=-1).T
    timing_total = (timing[:, 1:] - timing[:, 0, np.newaxis]).T
    if aggregate:
        timing_label = timing_label.sum(axis=1)
        timing_total = timing_total.sum(axis=1)

    total = dict(zip(labels, timing_total * scale))
    share = dict(zip(labels, timing_label / timing_total[-1, np.newaxis]))
    return total, share, timing


def chained_capture(*funcs, commit):
    """Pass the log output from several functions to the specified callable."""
    *first, final = funcs
    if not callable(commit):
        return final

    @wraps(final)
    def _wrapper(*args, **kwargs):
        for fn in funcs:
            nfo, gx = fn(*args, **kwargs)
            if gx is not None and fn is not final:
                raise RuntimeError(f"Only {final} is allowed to return non-None `gx`.")
            commit(nfo)
        return nfo, gx

    return _wrapper


def forward(model, input, *, hx=None):
    x = model.features(input)
    x, hx = masked_rnn(model.core, x, hx, reset=input.fin)
    out = model.head(x)
    return ValPolPair(out["val"].squeeze(-1), out["pol"].log_softmax(-1)), hx


def step(model, input, *, hx=None, nfo=None, deterministic=False):
    # breakpoint()
    vp, hx = forward(model, input, hx=hx)

    if deterministic:
        act_ = vp.pol.argmax(-1)

    else:
        act_ = multinomial(vp.pol.exp())

    return act_, vp, hx


def evaluate(input, output, hxx=None, nfo=None, *, epx, fill=True):
    # (sys) ignore the overlapping records between the fragment
    fragment = plyr.apply(lambda x: x[:-1], input)

    metrics = []
    # (log) collect the length and the accumulated rewards for each episode
    # XXX non-zero test via `fin != 0` is compatible with bool
    for ep in epx.extract(fragment.fin, fragment):
        ep_len = len(ep.fin) - int(ep.fin[-1] != 0)
        if ep_len < 1:
            continue

        metrics.append(
            {
                "f_len": int(ep_len),
                "f_ret": float(ep.rew[1:].sum()),
            }
        )

    if not metrics and fill:
        metrics = [{"f_len": np.nan, "f_ret": np.nan}]

    return {"metrics": metrics}, None


def prepare(model, input, output, hxx=None, nfo=None):
    # (sys) recompute the value-to-go estimates for truncated episodes, and
    #  adjust the rewards accordingly
    # XXX the truncated observations could be useful for all sort learnable
    #  objects, such as the env model in mu_\zero's MCTS, the IDM or curiosity
    rew_, fin_ = input.rew, input.fin
    mask, buf_, hx_ = extract_truncated(input, hxx=hxx, nfo=nfo)
    if mask is not None:
        with torch.no_grad():
            mu, _ = forward(model, buf_, hx=hx_)

        # (sys) scatter the re-computed value estimates
        # XXX same goes for `mu.gam` when applicable
        val_ = plyr.apply(torch.zeros_like, rew_)
        val_ = plyr.apply(lambda x, s: x.masked_scatter_(mask, s), val_, mu.val)

        # (sys) compute $r^\circ_t$ using the correct $v_t$ and $\gamma_t$
        # XXX we cannot change the reward data in the `rew` field inplace,
        #  since it might have been used as input to the agent
        _, _, rew_ = plyr.iapply(gamma, rew_, config.f_gam, val_, fin=fin_)

        # (sys) disable reward adjustment logic in `.utils.rl.returns.gamma()`
        #  by regressing from the ternary signal to boolean
        fin_ = fin_.ne(0)

    # use inverted (transposed) apply to fetch the returns and GAE
    ret, gae = plyr.iapply(pyt_ret_gae, rew_, output.val, config.f_gam, f_lam, fin=fin_)
    # XXX what if `rew` and `val` are structured? or the model outputs `gamma`
    #  (the price of rewards at `t` in terms of rewards at `t-1`:
    #  v_{t-1} <<-- r_t + \gamma_t v_t, \gamma_t ~ z_t, i.e. if the cumulative
    #  return is to be transferred back in time... term structure?)

    # (ppo) time-t synchronize the experience data (t=0..T-1)
    # XXX for H = 1 get (z_{t+j}, y_{t+j}, z_{t+j+1}, A_{t+j}, R_{t+j})_{j < H}
    curr_inp, curr_out = plyr.apply(lambda x: x[:-1], (input, output))
    next_inp = plyr.apply(lambda x: x[1:], input)
    return Buffer(  # .input, .out, .nex, .adv, .ret
        # z_t
        curr_inp,
        # y_t,
        curr_out,
        # z_{t+1}
        next_inp,
        # A_t depends on (r_s, \gamma_s)_{s > t} and all (v_s)_{s \geq t}
        gae,
        # R_t depends on (r_s, \gamma_s)_{s > t} and a bootstrap v_s, s > t
        ret,
    )


def sample(buffer, hxx=None, *, n_epochs, n_batch_size, n_seq=1):
    assert isinstance(buffer, Buffer)

    # (ppo) flatten the time-env dimensions
    flat = plyr.apply(torch.flatten, buffer, start_dim=0, end_dim=1)

    # (ppo) do several complete passes over the rollout buffer in batches
    n_size, n_envs = buffer.input.fin.shape
    offset = torch.arange(n_seq).mul_(n_envs).reshape(-1, 1)  # c-order seq x env

    # (sys) if the model is recurrent, then cat the runtimes along dim=1 to flatten
    hxx_getitem = None
    if isinstance(hxx, (list, tuple)) and hxx[-1] is not None:
        # (sys) deal with the only-once None in hxx[0] by ignoring the first
        #  input-hx pair if it is `None`
        if hxx[0] is None:
            hxx, n_size = hxx[1:], n_size - 1
            offset += n_envs

        # (sys) get a T * B tuple of runtimes, unbound along their batch `dim=1`
        # XXX by transposed application of `.unbind` we end up with the structure
        #  of the return value OUTSIDE of the nesting structure of the arguments.
        #  This way we get a `B x T x {...}` nesting instead of `T x {... x B}`.
        # XXX measurements have shown that using a numpy array of `object` (for
        #  advanced indexing and to avoid triggering dunder-array protocol) is
        #  MUCH slower than the basic list comprehensions with `.tolist()` and
        #  chain-zip generator
        hxx_getitem = tuple(
            chain(*zip(*plyr.iapply(torch.unbind, hxx, dim=1)))
        ).__getitem__

    for ep in range(n_epochs):
        # sample start time-env indices from t=0..T-L for all sub-sequences
        indices = torch.randperm((n_size - n_seq + 1) * n_envs)

        # (ppo) get a random batch of experience sub-sequences from the fragment
        hx = None
        for bi, start in enumerate(indices.split(n_batch_size)):
            if hxx_getitem is not None:
                hx_ = map(hxx_getitem, start.tolist())
                hx = plyr.apply(torch.stack, *hx_, _star=False, dim=1)

            batch = offset + start  # add next sub-sequence indices
            yield ep, bi, plyr.apply(lambda x: x[batch], flat), hx


def update_ppo(model, input, output, hxx=None, nfo=None):
    """PPO"""

    log = []
    buffer = prepare(model, input, output, hxx=hxx, nfo=nfo)

    # (ppo) train on random batches from experience
    for _, _, batch, hx in sample(
        buffer,
        hxx,
        n_epochs=config.n_ppo_epochs,
        n_batch_size=config.n_ppo_batch_size,
        n_seq=config.n_seq,
    ):
        # (ppo) diffable forward pass thru the random batch
        mu, _ = forward(model, batch.input, hx=hx)

        # (ppo) compute the importance weights (diffable)
        # XXX what if `act`, `pol`, `adv`, `ret` and `val` are structured?
        lik = torch.exp(
            rl.bselect(mu.pol, batch.nex.act, -1)
            - rl.bselect(batch.out.pol, batch.nex.act, -1)
        )
        f_clip = torch.abs(lik - 1).ge(config.f_ppo_eps).float().mean()

        # detail #5: for value targets use TD(\lambda) returns from GAE
        ret = batch.adv + batch.out.val  # plyr.apply(op.add, .adv, .out.val)

        # detail #7: advantage normalization
        adv = batch.adv
        if config.b_adv_normalization:
            # Although we can safely subtract baseline const in expected policy
            # gradients, it might not be safe in sampled approximation.
            # XXX conceptually, normalization should keep the signs, e.g.
            #  div-by-norm, to  however early on the baseline could be bad
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # (ppo) the clipped policy grad objective (equivalently simplified)
        # for `x > 0` we have `\min\{r A, clip(r, 1-x, 1+x) A \} -->> \max`
        #     if `A > 0` then `A \min\{r, 1+x \} = \min\{r A, A + x A \}`
        #     if `A < 0` then `A \max\{r, 1-x \} = \min\{r A, A - x A \}`
        # note, `A + A x` for `A > 0` and `A - A x` for `A < 0` is `A + x |A|`
        terms = {
            "polgrad": torch.minimum(
                adv * lik, adv + abs(adv) * config.f_ppo_eps
            ).mean(),
            "entropy": rl.entropy(mu.pol, -1).mean(),
            "critic": F.mse_loss(mu.val, ret, reduction="mean"),
        }

        optim.zero_grad()
        reduce(terms, config.f_loss_coef).backward()
        clip_grad_norm_(model.parameters(), config.f_clip_grad)
        optim.step()

        log.append(plyr.apply(float, {**terms, "f_clip": f_clip}))

    # (sys) recompute the recurrent state `hx` AFTER the update
    # XXX `hx` is $h_T$ from $h_{t+1}, y_t = F(z_t, h_t; w)$, t=0..T-1.
    # XXX A side effect of training is that the recurrent runtime state in
    #  `hxx[-1]` has become stale, i.e. it no longer corresponds to the final
    #  `hx` had it been computed by the updated policy on the same fragment.
    #  Idealy, we would like to recompute `hx` over the entire historical
    #  trajectory, had we stored it whole. The next best option is to make
    #  a second pass over the fragment $
    #      (z_t)_{t=0}^{T-1}
    #  $ and use $h'_T$ as the new `hx`, where $
    #      h'_{t+1}, y_t = F(z_t, h'_t; w')
    #  $, t=0..T-1, and $h'_0 = h_0$ is given by `gx` in `hxx[0]`.
    hx = hxx[-1]
    if hx is not None:
        fragment = plyr.apply(lambda x: x[:-1], input)
        with torch.no_grad():
            _, hx = forward(model, fragment, hx=hxx[0])

    return {"update": log}, hx


class NonRNN(nn.Module):
    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    def forward(self, input, hx=None):
        assert hx is None
        return self.body(input), hx


def build_model(*, linear=nn.Linear, tanh=nn.Tanh, **ignored):
    # architectures that seem to have worked well
    # - emb(obs)_{64} || emb(act)_{64} -> Rearrange (2x64) -> LayerNorm(64)
    # - linear(obs; param=emb(act)) <<-- hypernetwork
    # generally have not worked
    # - double-sized single shared network with split heads
    # - shared previous action embedding

    # XXX cuda, apparently, is not very keen on zero-sized embeddings
    act = nn.Embedding(4, config.n_act_embedding_dim)
    obs = linear(8, 64)
    n_features = 64 + config.n_act_embedding_dim
    core = NonRNN(nn.Identity())

    if config.b_recurrent:
        obs = nn.Sequential(obs, tanh())
        core = nn.GRU(n_features, 64)
        n_features = 64

    # split policy-value networks
    pol = nn.Sequential(
        InputGradScaler(scale=0.1),
        tanh(),
        linear(n_features, 64),
        tanh(),
        linear(64, 4),
    )

    val = nn.Sequential(
        tanh(),
        linear(n_features, 64),
        tanh(),
        linear(64, 1),
    )

    return nn.ModuleDict(
        dict(
            # shared feature embedding
            features=ModuleDict(dict(obs=obs, act=act), dim=-1),
            # shared neural recurrent core
            core=core,
            # split networks
            head=ModuleDictSplitter(dict(pol=pol, val=val)),
        )
    )


@torch.no_grad()
def parameters_lerp(frac, *, src, dst):
    # we do not check if the models are compatible!
    for s, d in zip(src.parameters(), dst.parameters()):
        d.lerp_(s, frac)  # d += frac * (s - d)


if __name__ == "__main__":
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    b_visualize: bool = True
    n_eval_episodes: int = 99

    # ad-hoc logging
    to_seconds = EngFormatter(places=2, sep="", unit="s")
    to_number = EngFormatter(places=1, sep="", unit="")
    n_iters, history, timings_ns = 0, [], []
    header, format = map(
        " ".join,
        zip(
            *(
                ("iter", "{:>4d}"),
                ("n_steps ", "{:>8d}"),
                # "update"
                ("polgrad ", "{polgrad:+8.1e}"),
                ("entropy ", "{entropy:<8.3f}"),
                ("critic  ", "{critic:8.2e}"),
                ("f_clip", "{f_clip:<6.2f}"),
                ("f_lam", "{f_lam:<5.2f}"),
                # "metrics"
                ("f_len", "{f_len:<5.0f}"),
                ("f_ret ", "{f_ret:<+6.1f}"),
                # timings
                ("f_cap", "{f_cap:<5.2f}"),
                ("loop", "{s_ns_loop}"),
            )
        ),
    )
    print(config)
    print()
    print(header)

    # seprate models for stepping/learning and evaluation
    model_evaluation = build_model()

    model_stepper = build_model().to(device_)
    model_learner = build_model().to(device_)
    parameters_lerp(1.0, src=model_learner, dst=model_stepper)

    optim = torch.optim.Adam(
        model_learner.parameters(),
        lr=config.f_lr,
        eps=config.f_optim_eps,
    )

    epx = EpisodeExtractor()
    env = rl.SerialVecEnv(
        gym.make,
        config.n_envs,
        args=("LunarLander-v2",),
        ternary=config.b_ternary,
    )

    log = {}
    cap = buffered(
        partial(step, model_stepper, deterministic=False),
        chained_capture(
            partial(evaluate, epx=epx, fill=True),
            partial(update_ppo, model_learner),
            commit=log.update,
        ),
        config.n_fragment_length,
        device=device_,
    )

    # the number of steps taken in all envs so far
    n_steps, ns_total = 0, monotonic_ns()

    # the main loop: `prepare launch (step send)*`
    act = launch(cap, rl.prepare(env).npy)  # XXX a dedicated `.prepare`?
    while n_steps < config.n_total_steps:
        # (train) get GAE averaging schedule
        f_lam = config.c_lam_anneal(n_steps / config.n_total_steps)

        timings_ns.append(monotonic_ns())  # base

        result = env.step(act)  # XXX High number of collisions slows down Box2D
        timings_ns.append(monotonic_ns())  # env.step

        act = cap.send(result)  # XXX cap updates `act` INPLACE, no lulz here :)
        timings_ns.append(monotonic_ns())  # cap.send

        n_steps += config.n_envs
        if log:
            n_iters += 1

            # (log) average the logged numeric data (RL loss and metrics)
            terms = plyr.apply(np.mean, *log["update"], _star=False)
            metrics = plyr.apply(np.mean, *log["metrics"], _star=False)
            log.clear()

            # (log) collect the timings and get the fraction spend at each step
            n_total, f_share, timings = process_timings(
                timings_ns,
                "f_env",
                "f_cap",
                scale=1e-9,
                aggregate=True,
            )
            history.append((timings, n_steps, metrics, terms))
            timings_ns.clear()

            # (log) format and print
            print(
                format.format(
                    n_iters,
                    n_steps,
                    f_lam=f_lam,
                    s_ns_loop=to_seconds(n_total["f_cap"]),
                    **terms,
                    **f_share,
                    **metrics,
                )
            )

            # stepper follows the learner as EWMA with decay rate `f_model_lerp`
            parameters_lerp(config.f_model_lerp, src=model_learner, dst=model_stepper)

    # (sys) close the capsule and flush the peisode collector
    cap.close()
    epx.finish()

    # show the config, timings and reward dynamics
    ns_total = monotonic_ns() - ns_total
    print()
    print(config)
    print(model_stepper)
    print(f"{to_number(n_steps)} steps took {to_seconds(ns_total * 1e-9)}.")

    # load the stepper into the evaluation model
    state_dict = model_stepper.state_dict()
    print(model_evaluation.load_state_dict(state_dict))

    if b_visualize:
        timings, n_steps, metrics, logs = plyr.apply(np.asarray, *history, _star=False)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), dpi=120)

        # timings
        n_total, f_share, _ = process_timings(
            timings.flatten(),
            "f_env",
            "f_cap",
            aggregate=False,
        )
        for nom in ("f_cap",):
            ax0.plot(np.r_[: n_steps[-1] : config.n_envs], f_share[nom], label=nom)

        # mark the `update` events on the time axis
        for x in n_steps:
            ax0.axvline(x, c="k", alpha=0.25, zorder=-10)
        ax0.legend()

        # performance
        (l_f_len,) = ax1.plot(n_steps, metrics["f_len"], c="C1", label="length")

        ax_ = ax1.twinx()
        (l_f_ret,) = ax_.plot(n_steps, metrics["f_ret"], c="C0", label="return")

        ax1.legend(loc="best", handles=[l_f_ret, l_f_len])

        plt.tight_layout()
        plt.show()

    # visualized evaluation runs
    env = rl.SerialVecEnv(
        gym.make,
        1,
        args=("LunarLander-v2",),
        ternary=config.b_ternary,
    )
    vis = env.envs[0].render if b_visualize else lambda: True

    history = []
    cap = buffered(
        partial(step, model_evaluation, deterministic=True),
        chained_capture(
            partial(evaluate, epx=epx, fill=False),
            commit=lambda nfo: history.extend(nfo["metrics"]),
        ),
        config.n_fragment_length,
    )

    act = launch(cap, rl.prepare(env).npy)
    while len(history) < n_eval_episodes and vis():
        act = cap.send(env.step(act))

    cap.close()
    epx.finish()

    # get the evaluation averages
    if history:
        avg = plyr.apply(np.mean, *history, _star=False)
        std = plyr.apply(np.std, *history, _star=False)
        print(plyr.apply("{:4.1f}±{:4.1f}".format, avg, std))
