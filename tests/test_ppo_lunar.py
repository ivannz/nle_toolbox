import plyr
import torch
import numpy as np

import operator as op
from functools import wraps, partial
from collections import namedtuple

from time import monotonic_ns

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from nle_toolbox.utils.nn import ModuleDict, multinomial, ModuleDictSplitter

import gym
import nle_toolbox.utils.rl.engine as rl
from nle_toolbox.utils.rl.capsule import buffered, launch
from nle_toolbox.utils.rl.returns import pyt_ret_gae

from nle_toolbox.utils.rl.tools import EpisodeExtractor
from dataclasses import dataclass

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
    n_total_steps: int = 1_000_000
    n_fragment_length: int = 1024
    n_envs: int = 16

    f_gam: float = 0.999
    n_ppo_batch_size: int = 64
    n_ppo_epochs: int = 4

    f_clip_grad: float = 0.5
    f_ppo_eps: float = 0.2
    n_act_embedding_dim: int = 16
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
Buffer = namedtuple("Buffer", "input,act,log_p,adv,ret")


def reduce(values, weight=None):
    flat = []
    if weight is not None:
        values = plyr.apply(op.mul, values, weight)

    plyr.apply(flat.append, values)
    return sum(flat)


def timing_eng(ns, units=("ns", "µs", "ms", "s.")):
    # ns -->> µs -->> ms -->> s.
    for _unit in units:
        if ns < 1000:
            break
        ns /= 1000
    return f"{ns:.1f}{_unit}"


def capture(fn, to):
    """Capture the log information output to the specified dict."""
    if to is None:
        return fn

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        nfo, _ = result = fn(*args, **kwargs)
        to.update(nfo)
        return result

    return _wrapper


def entropy(logprob):
    # `rl.entropy` relies on synchronisation of `logprob`
    return (
        F.kl_div(
            logprob.new_zeros(()),
            logprob,
            reduction="none",
            log_target=True,
        )
        .sum(dim=-1)
        .neg()
    )


def pyt_logpact(logpol, act):
    # `rl.pyt_logpact` relies on synchronisation of `logprob` with `act`
    return logpol.gather(-1, act.unsqueeze(-1)).squeeze_(-1)


def step(model, input, *, hx=None, nfo=None):
    # breakpoint()

    out = model(input)
    vp = ValPolPair(out["val"].squeeze(-1), out["pol"].log_softmax(-1))
    act_ = multinomial(vp.pol.exp())
    return act_, vp, None


def eval_step(model, obs, act=None, rew=None, fin=None, *, hx=None):
    return step(model, rl.Input(obs, act, rew, fin), hx=hx, nfo=None)


def ep_metrics(epx, fragment):
    metrics = []
    for ep in epx.extract(fragment.fin, fragment):
        ep_len = len(ep.fin) - int(ep.fin[-1])
        if ep_len < 1:
            continue

        metrics.append(
            (
                ep_len,
                ep.rew[1:].sum(),
            )
        )

    if not metrics:
        return {}

    f_len, f_ret = map(np.mean, zip(*metrics))
    return {"f_len": f_len, "f_ret": f_ret}


def prepare(input, output, nfo=None):
    # XXX what if `act`, `rew` and `val` are structured?
    # breakpoint()

    _, gae = pyt_ret_gae(
        input.rew[1:],
        input.fin[1:],
        output.val,
        gam=config.f_gam,
        lam=f_lam,
    )

    return Buffer(
        plyr.apply(lambda x: x[:-1], input),
        input.act[1:],
        rl.pyt_logpact(output.pol, input.act),
        gae,
        # detail #5: TD(\lambda) returns with gae
        gae + output.val[:-1],
    )


def sample(buffer, n_epochs, n_batch_size):
    # (ppo) flatten the time-env dimensions
    flat = plyr.apply(torch.flatten, buffer, start_dim=0, end_dim=1)

    # (ppo) do several complete passes over the rollout buffer in batches
    for _ in range(n_epochs):
        indices = torch.randperm(flat.input.fin.numel())

        # (ppo) get a random batch from the experience fragment
        for batch in indices.split(n_batch_size):
            yield plyr.apply(lambda x: x[batch], flat)


def update_ppo(model, epx, input, output, gx=None, hx=None, nfo=None):
    """PPO"""
    buffer = prepare(input, output, nfo=nfo)

    log = []
    # (ppo) train on random batches from experience
    for batch in sample(buffer, config.n_ppo_epochs, config.n_ppo_batch_size):
        # (ppo) diffable forward pass thru the random batch
        out = model(batch.input)  # , hx=hx, nfo=nfo)
        mu = ValPolPair(out["val"].squeeze(-1), out["pol"].log_softmax(-1))

        # (ppo) compute the importance weights (diffable)
        lik = torch.exp(pyt_logpact(mu.pol, batch.act) - batch.log_p)
        f_clip = torch.abs(lik - 1).ge(config.f_ppo_eps).float().mean()

        # detail #7: advantage normalization
        adv = (batch.adv - batch.adv.mean()) / (batch.adv.std() + 1e-8)
        terms = {
            "polgrad": torch.minimum(
                adv * lik,
                adv * lik.clamp(1.0 - config.f_ppo_eps, 1.0 + config.f_ppo_eps),
            ).mean(),
            "entropy": entropy(mu.pol).mean(),
            "critic": F.mse_loss(mu.val, batch.ret, reduction="mean"),
        }

        optim.zero_grad()
        reduce(terms, config.f_loss_coef).backward()
        clip_grad_norm_(model.parameters(), config.f_clip_grad)
        optim.step()

        log.append(plyr.apply(float, {**terms, "f_clip": f_clip}))

    # (log) average the rl loss components
    log = plyr.apply(np.mean, *log, _star=False)

    # (log) estimate the number of resets per fragment
    # XXX `batch.input` has correct sync!
    metrics = ep_metrics(epx, buffer.input)
    if metrics:
        log["metrics"] = metrics

    return log, None


if __name__ == "__main__":
    model_stepper = model_learner = nn.Sequential(
        # shared feature embedding
        ModuleDict(
            dict(
                obs=nn.Sequential(
                    nn.Linear(8, 64),
                    nn.Tanh(),
                ),
                act=nn.Embedding(4, config.n_act_embedding_dim),
            ),
            dim=-1,
        ),
        # split networks
        ModuleDictSplitter(
            dict(
                pol=nn.Sequential(
                    nn.Linear(64 + config.n_act_embedding_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 4),
                ),
                val=nn.Sequential(
                    nn.Linear(64 + config.n_act_embedding_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1),
                ),
            )
        ),
    )

    optim = torch.optim.Adam(
        model_learner.parameters(),
        lr=3e-4,
        eps=1e-5,  # detail #?: it turns out, this is very important!
    )

    env = rl.SerialVecEnv(
        gym.make,
        config.n_envs,
        args=("LunarLander-v2",),
        pack_nfo=False,
    )

    log = {}
    epx = EpisodeExtractor()
    cap = buffered(
        partial(step, model_stepper),
        capture(
            partial(update_ppo, model_learner, epx),
            log,
        ),
        config.n_fragment_length,
    )

    # ad-hoc logging
    history = []
    metrics = {"f_len": np.nan, "f_ret": np.nan}
    header, format = map(
        " ".join,
        zip(
            *(
                ("iter", "{:>4d}"),
                ("n_steps ", "{:>8d}"),
                ("polgrad ", "{polgrad:+8.1e}"),
                ("entropy ", "{entropy:<8.3f}"),
                ("critic  ", "{critic:8.2e}"),
                ("f_clip", "{f_clip:<6.2f}"),
                ("f_len", "{f_len:<5.0f}"),
                ("f_ret ", "{f_ret:<+6.1f}"),
                ("f_lam", "{f_lam:<5.2f}"),
                ("f_cap", "{f_cap:<5.2f}"),
                ("loop", "{s_ns_loop}"),
            )
        ),
    )
    print(header)

    n_steps, n_iters = 0, 0
    timings_ns = []
    act = launch(cap, rl.prepare(env).npy)  # XXX a dedicated `.prepare`?
    while n_steps < config.n_total_steps:
        # (train) get GAE averaging schedule
        f_lam = config.c_lam_anneal(n_steps / config.n_total_steps)

        timings_ns.append(monotonic_ns())  # base

        result = env.step(act)  # XXX High number of collisions slows down Box2D
        timings_ns.append(monotonic_ns())  # env.step

        act = cap.send(result)  # XXX cap updates `act` INPLACE, no lulz here :)
        timings_ns.append(monotonic_ns())  # cap.step

        n_steps += config.n_envs

        if log:
            n_iters += 1

            # collect the timings and get the fraction spend at each step
            timings = np.reshape(timings_ns, (-1, 3), "C")  # XXX 3 = 1 + N

            ns_step = np.diff(timings, axis=-1).sum(0)
            ns_loop = (timings[:, -1] - timings[:, 0]).sum(0)
            f_share = dict(
                zip(
                    (
                        "f_env",
                        "f_cap",
                    ),
                    ns_step / ns_loop,
                )
            )
            timings_ns.clear()

            # metrics are sparsely logged, so we repeat the most recently
            #  available values
            metrics = log.pop("metrics", metrics)

            print(
                format.format(
                    n_iters,
                    n_steps,
                    **log,
                    f_lam=f_lam,
                    **f_share,
                    s_ns_loop=timing_eng(ns_loop),
                    **metrics,
                )
            )

            log.clear()

    print(config)

    eval_env = rl.SerialVecEnv(gym.make, 1, args=("LunarLander-v2",))
    stepper = partial(eval_step, model_stepper)
    with torch.no_grad():
        npyt = rl.prepare(eval_env)

        vis = eval_env.envs[0]
        while vis.render():
            (inp, out), _, _ = rl.step(eval_env, stepper, npyt, hx=None)
