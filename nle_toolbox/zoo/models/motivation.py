import math

import torch
from torch import nn
from torch.nn import functional as F

from collections import namedtuple

from .basic import SimpleEncoder
from ...utils.nn import trailing_lerp

MotivatorObs = namedtuple("MotivatorObs", "agent,obs")


class RNDNetwork(nn.Module):
    """The function, which embeds the observations for RND."""

    def __init__(self, obs: dict, act: dict, sizes: list[int]) -> None:
        super().__init__()

        # the core embedding modules (observation and action)
        self.obs = SimpleEncoder(**obs)
        self.act = nn.Embedding(**act) if act is not None else None
        n_features = obs["intermediate_size"] + (
            act["embedding_dim"] if act is not None else 0
        )

        layers = [nn.Linear(n_features, sizes[0], bias=True)]
        for n, m in zip(sizes, sizes[1:]):
            layers.append(nn.GELU())
            layers.append(nn.Linear(n, m, bias=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs, act=None):
        x = self.obs(obs)
        if self.act is not None:
            x = torch.cat((x, self.act(act)), dim=-1)

        return self.encoder(x)

    @staticmethod
    def default_recipe(
        embedding_dim: int = 16,
        intermediate_size: int = 256,
        sizes: tuple[int] = (256,),
        *,
        k: int = 3,
        bls: tuple[str] = (
            "hp",
            "hunger",
            "condition",
        ),
        act: dict = None,
    ) -> dict:
        return {
            # symmetric ego-centric window of glyphs plus bottom line stats
            "obs": {
                "n_context": (k + 1 + k) * (k + 1 + k) + len(bls),
                "embedding_dim": embedding_dim,
                "intermediate_size": intermediate_size,
                "dropout": 0.0,
            },
            "act": act,
            "sizes": sizes,
        }


RNDNetwork_recipe = RNDNetwork.default_recipe()


class RNDModule(nn.Module):
    r"""Exploration rewards via Random Network Distillation.

    Details
    -------
    The key idea of RND is to employ the training process of the online network
    as a way to memoize the experienced observations and indirectly gradually
    decay their novelty, which is tied to the prediction error against a fixed
    random target model. In a certain sense the online network strives to learn
    the stationary density of the encountered observations. After each $
        t \to t+1
    $ transition the RND motivator computes non-diffable intrinsic reward with
    $$
    (x_t, a_t, x_{t+1})
        \mapsto r^I_t = \ell\bigl(
            f(x_{t+1}),
            \overline{f}(x_{t+1})
        \bigr)
        \,
        $$
    where the prediction error between embeddings $u$ and $v$ is $
        \ell(u, v) = \frac1d \|u - v\|_2^2
    $ and $
        f \colon \mathcal{X} \to \marhbb{R}^d
    $ is the m-dim observation embedder ($\overline{f}$ is the target network).

    The original paper uses the mean squared error for the loss, but denotes it
    by the `squared norm` (Burda et al. (2018) sec. 2.2) and the rewards (ibid,
    sec. A.2). At the same time the code base of RIDE (Raileanu and Rocktäschel
    (2020)) uses ell-2 norms, i.e. the square root of the MSE.

    We argue, that 1) the intrinsic rewards should not depend on the embedding
    dim and average across it, for otherwise "wider" spaces wouldn't be inter-
    comparable with smaller ones, and 2) using the norms protects against large
    residuals, which destabilize and screen the extrinsic rewards, since $
       \partial_r \|r\|_2^2 = 2 r
    $ while $
       \partial_r \|r\|_2 = \frac{r}{\|r\|_2}
    $.
    """

    def __init__(
        self,
        embed: dict,
        *,
        root: bool = False,
        **ignore,
    ) -> None:
        super().__init__()

        self.root = root

        # the non-learnable target and the learnable online networks
        self.target = RNDNetwork(**embed).requires_grad_(False).eval()
        self.online = RNDNetwork(**embed).requires_grad_(True).train()

        # RND does not have a runtime state, other than the networks' params
        self.register_parameter("h0", None)

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        assert isinstance(obs, MotivatorObs) and hx is None

        # get the target and online embeddings
        online = self.online(obs.obs, obs.agent)
        with torch.no_grad():
            target = self.target(obs.obs, obs.agent)

        # get dimensionless novelty rewards
        if not self.root:
            err = torch.norm(online - target, p=2, dim=-1)
            rew = err.detach() / math.sqrt(target.shape[-1])

        else:
            err = F.mse_loss(online, target, reduction="none").sum(-1)
            rew = err.detach() / target.shape[-1]

        # return the rewards and the errors
        return rew, err, None

    @classmethod
    def default_recipe(
        cls: type,
        embed: dict = RNDNetwork_recipe,
        *,
        root: bool = False,
    ) -> dict:
        return {
            "cls": str(cls),
            "embed": embed,
            "root": root,
        }


RNDModule_recipe = RNDModule.default_recipe()


class RIDEEmbedding(nn.Module):
    def __init__(
        self,
        obs: dict,
        *,
        h0: bool = True,
    ):
        super().__init__()

        # build the core embedding module
        self.embedding = SimpleEncoder(**obs)

        # set up learnable embedding of the terminal observation, which should
        # have the from `T x B x ...`
        if h0:
            self.h0 = nn.Parameter(torch.randn(1, 1, obs["intermediate_size"]))

        else:
            self.register_parameter("h0", None)

        self.out_features = obs["intermediate_size"]

    def forward(self, obs, fin=None, *, hx=None):
        r"""
        Details
        -------
        RIDE motivator uses the stored last observation's embedding
        $\xi_{t-1}$ and the current observation $x_t$ and whether it is
        terminal ($f_t = \top$) to assess the last transition's impact on
        ENV ($t-1 \to t$ under $a_{t-1}$) and determine the appropriate
        reward $
            r^I(\xi_{t-1}, \tilde{\xi_t})
        $ with $
            \xi_t = \phi(x_t)
        $ and $
            \tilde{\xi}_t = (1 - f_t) \xi_t + f_t \bar{\xi}
        $.

        The current implementation repurposes `hx` to store $\xi_{t-1}$, but
        the motivator does not conform to the resetting logic of `masked_rnn`.
        The state `hx` is never affected by the reset flag and, instead, the
        current observation's embedding is TEMPORARILY set to the terminal
        state embedding for the purpose of computing the intrinsic reward.
        """
        xi_ = self.embedding(obs)

        # prepare the embeddings of the previous and the terminal observations
        #  `hx` and `h0`, respectively
        h0 = self.h0  # XXX `h0` defaults to zero, but could be learnable
        if h0 is not None and hx is None:
            # no previous embedding $\xi_{t-1}$ is available, but we've got
            #  learnable terminal embedding `h0`: treat it as the `observation
            #  before existence` and init `hx` to it
            hx = torch.cat((h0,) * xi_.shape[1], dim=1)

        elif h0 is None:
            # default the terminal embedding to zero and init the previous
            #  embedding $\xi_{t-1}$, if it is missing
            if hx is None:
                hx = torch.zeros_like(xi_[:1])

            h0 = hx.new_zeros(())  # init to scalar zero

        # use `hx` to build the series of lagged embeddings (for $x_{t-1}$)
        # XXX $\xi_t = \phi(x_t)$, t=0..T-1, with \xi_{-1} given by `hx`
        if len(xi_) > 1:
            xi0 = torch.cat((hx, xi_[:-1]), dim=0)

        else:
            xi0 = hx

        # if $x_t$ is terminal TEMPORARILY replace its embedding with that of
        # the terminal observation
        # XXX $r_t = d(\xi_{t-1}, \tilde{xi}_t)$ with $\tilde{\xi}_t$ being
        # $\xi_t$ if $f_t = \top$ or $\bar{\xi}$ if $f_t$ is $\bot$.
        if fin is not None:
            xi1 = trailing_lerp(xi_, h0, eta=fin.to(xi_), leading=0)
            # XXX `xi1 <<-- xi_ * (1 - f) + h0 * f`

        else:
            xi1 = xi_

        # the next `hx` is $\xi_t$ and NOT $\tilde{\xi}_t$
        return (xi0, xi1), xi_[-1:]

    @staticmethod
    def default_recipe(
        embedding_dim: int = 16,
        intermediate_size: int = 256,
        *,
        k: int = 3,
        bls: tuple[str] = (
            "hp",
            "hunger",
            "condition",
        ),
        h0: bool = True,
    ) -> dict:
        return {
            # symmetric ego-centric window of glyphs plus bottom line stats
            "obs": {
                "n_context": (k + 1 + k) * (k + 1 + k) + len(bls),
                "embedding_dim": embedding_dim,
                "intermediate_size": intermediate_size,
                "dropout": 0.0,
            },
            "h0": h0,
        }


RIDEEmbedding_recipe = RIDEEmbedding.default_recipe()


class CatLinear(nn.Linear):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in1_features + in2_features, out_features, bias=bias)

    def forward(self, inputs: tuple[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=-1))


class Bilinear(nn.Bilinear):
    def forward(self, inputs: tuple[torch.Tensor]) -> torch.Tensor:
        return super().forward(*inputs)


class RIDEModule(nn.Module):
    def __init__(
        self,
        obs: dict,
        act: dict,
        *,
        sizes: list[int],
        h0: bool = True,
        bilinear: bool = False,
        flip: bool = False,
    ):
        assert len(sizes) == 1
        super().__init__()

        # the core embedding modules (observation and action)
        self.embedding = RIDEEmbedding(obs, h0=h0)
        self.act = nn.Embedding(**act)

        # setup the forward and inverse models
        self.fwd = nn.Sequential(
            (Bilinear if bilinear else CatLinear)(
                obs["intermediate_size"],
                act["embedding_dim"],
                sizes[0],
            ),
            nn.GELU(),
            nn.Linear(sizes[0], obs["intermediate_size"]),
        )

        self.inv = nn.Sequential(
            CatLinear(
                obs["intermediate_size"],
                obs["intermediate_size"],
                sizes[0],
            ),
            nn.GELU(),
            nn.Linear(sizes[0], act["num_embeddings"]),
        )

        self.flip = flip

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        assert isinstance(obs, MotivatorObs)
        (xi0, xi1), hx = self.embedding(obs.obs, fin, hx=hx)

        # issue the non-diffable intrinsic reward for the transition
        #        x_{t-1}, a_{t-1} -->> x_t
        rew = torch.norm(xi0.detach() - xi1.detach(), p=2, dim=-1)
        rew.div_(math.sqrt(xi1.shape[-1]))  # XXX in-place!
        # XXX scale by the root of the embedding dim to make rewards
        #  dimensionless, since $\|v\|_2 \leq \sqrt{d} \|v\|_\infty$

        if self.flip:
            xi0, xi1 = xi1, xi0
        return rew, (xi0, xi1), hx

    @classmethod
    def default_recipe(
        cls: type,
        n_actions: int,
        embed: dict = RIDEEmbedding_recipe,
        sizes: tuple[int] = (256,),
        *,
        bilinear: bool = False,
        flip: bool = False,
    ) -> dict:
        embedding_dim = embed["obs"]["embedding_dim"]
        return {
            "cls": str(cls),
            **embed,
            "sizes": sizes,
            "act": {
                "num_embeddings": n_actions,
                "embedding_dim": embedding_dim,
            },
            "bilinear": bilinear,
            "flip": flip,
        }
