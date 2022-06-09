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

from nle_toolbox.utils.nn import multinomial
from nle_toolbox.utils.nn import ModuleDict, ModuleDictSplitter

import gym
import nle_toolbox.utils.rl.engine as rl
from nle_toolbox.utils.rl.capsule import buffered, launch
from nle_toolbox.utils.rl.returns import pyt_ret_gae

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
    n_total_steps: int = 1_000_000
    n_fragment_length: int = 1024
    n_envs: int = 16

    n_act_embedding_dim: int = 0

    f_gam: float = 0.999
    n_ppo_batch_size: int = 64
    n_ppo_epochs: int = 4

    f_ppo_eps: float = 0.2
    b_adv_normalization: bool = True
    f_clip_grad: float = 0.5
    f_entropy: float = 0.01
    f_critic: float = 0.5

    f_model_lerp: float = 0.9

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


def forward(model, input, *, hx=None, nfo=None):
    out = model(input)
    return ValPolPair(out["val"].squeeze(-1), out["pol"].log_softmax(-1))


def step(model, input, *, hx=None, nfo=None, deterministic=False):
    # breakpoint()
    vp = forward(model, input, hx=hx, nfo=nfo)

    if deterministic:
        act_ = vp.pol.argmax(-1)

    else:
        act_ = multinomial(vp.pol.exp())

    return act_, vp, None


def evaluate(input, output, gx=None, hx=None, nfo=None, *, epx, fill=True):
    # (sys) ignore the overlapping records between the fragment
    fragment = plyr.apply(lambda x: x[:-1], input)

    metrics = []
    # (log) collect the length and the accumulated rewards for each episode
    for ep in epx.extract(fragment.fin, fragment):
        ep_len = len(ep.fin) - int(ep.fin[-1])
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
        rl.bselect(output.pol[:-1], input.act[1:], -1),
        gae,
        # detail #5: TD(\lambda) returns with gae
        gae + output.val[:-1],
    )


def sample(buffer, n_epochs, n_batch_size):
    # (ppo) flatten the time-env dimensions
    flat = plyr.apply(torch.flatten, buffer, start_dim=0, end_dim=1)

    # (ppo) do several complete passes over the rollout buffer in batches
    for ep in range(n_epochs):
        indices = torch.randperm(flat.input.fin.numel())

        # (ppo) get a random batch from the experience fragment
        for bi, batch in enumerate(indices.split(n_batch_size)):
            yield ep, bi, plyr.apply(lambda x: x[batch], flat)


def update_ppo(model, input, output, gx=None, hx=None, nfo=None):
    """PPO"""
    buffer = prepare(input, output, nfo=nfo)

    log = []
    # (ppo) train on random batches from experience
    for _, _, batch in sample(buffer, config.n_ppo_epochs, config.n_ppo_batch_size):
        # (ppo) diffable forward pass thru the random batch
        mu = forward(model, batch.input)  # , hx=hx, nfo=nfo)

        # (ppo) compute the importance weights (diffable)
        lik = torch.exp(rl.bselect(mu.pol, batch.act, -1) - batch.log_p)
        f_clip = torch.abs(lik - 1).ge(config.f_ppo_eps).float().mean()

        # detail #7: advantage normalization
        adv = batch.adv
        if config.b_adv_normalization:
            # Although we can safely subtract baseline consts in expectated
            #  policy gradients, it might not be safe in sampled approximation.
            # XXX conceptually, normalization should keep the signs, e.g.
            #  div-by-norm, to  however early on the baseline could be bad
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        terms = {
            "polgrad": torch.minimum(
                adv * lik,
                adv * lik.clamp(1.0 - config.f_ppo_eps, 1.0 + config.f_ppo_eps),
            ).mean(),
            "entropy": rl.entropy(mu.pol, -1).mean(),
            "critic": F.mse_loss(mu.val, batch.ret, reduction="mean"),
        }

        optim.zero_grad()
        reduce(terms, config.f_loss_coef).backward()
        clip_grad_norm_(model.parameters(), config.f_clip_grad)
        optim.step()

        log.append(plyr.apply(float, {**terms, "f_clip": f_clip}))

    return {"update": log}, None


def build_model(*, linear=nn.Linear, tanh=nn.Tanh, **ignored):
    # architectures that seem to have worked well
    # - emb(obs)_{64} || emb(act)_{64} -> Rearrange (2x64) -> LayerNorm(64)
    # - linear(obs; param=emb(act)) <<-- hypernetwork
    # - double-sized single shared network with split heads doesn't work
    # - shared previous action embedding generally does not work
    return nn.Sequential(
        # shared feature embedding
        ModuleDict(
            dict(
                obs=nn.Identity(),
                # XXX cuda, apparently, does not enjoy zero-sized embeddings
                act=nn.Embedding(4, config.n_act_embedding_dim),
            ),
            dim=-1,
        ),
        # split networks
        ModuleDictSplitter(
            dict(
                pol=nn.Sequential(
                    linear(8 + config.n_act_embedding_dim, 64),
                    tanh(),
                    linear(64, 64),
                    tanh(),
                    linear(64, 4),
                ),
                val=nn.Sequential(
                    linear(8 + config.n_act_embedding_dim, 64),
                    tanh(),
                    linear(64, 64),
                    tanh(),
                    linear(64, 1),
                ),
            )
        ),
    )


@torch.no_grad()
def parameters_lerp(frac, *, src, dst):
    # we do not check if the models are compatible!
    for s, d in zip(src.parameters(), dst.parameters()):
        d.lerp_(s, frac)  # d += frac * (s - d)


if __name__ == "__main__":
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    b_visualize: bool = False
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
    print(header)

    # seprate models for stepping/learning and evaluation
    model_evaluation = build_model()

    model_stepper = build_model().to(device_)
    model_learner = build_model().to(device_)
    parameters_lerp(1.0, src=model_learner, dst=model_stepper)

    optim = torch.optim.Adam(
        model_learner.parameters(),
        lr=3e-4,
        eps=1e-5,  # detail #?: it turns out, this is very important!
    )

    epx = EpisodeExtractor()
    env = rl.SerialVecEnv(
        gym.make,
        config.n_envs,
        args=("LunarLander-v2",),
        pack_nfo=False,
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
    env = rl.SerialVecEnv(gym.make, 1, args=("LunarLander-v2",))
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
