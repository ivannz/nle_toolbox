import math

import torch
from torch import nn
from torch.nn import functional as F

from collections import namedtuple

from .vq import VectorQuantizedVAE, VQEOutput

# We need to:
# 1. collect commitment and embedding losses, regardless of their diffability
# 2. do this ONLY when we explicitly need to, without resorting to `.training`
#    or `.requires_grad` checks
# 3. implementing a mind-reading intention-predicting god-like layer is silly


class BaseVQHelper:
    """A context object, which tracks the VQ layers in the module."""

    def __init__(
        self,
        module: nn.Module,
        cls: type = VectorQuantizedVAE,
    ) -> None:
        # enumerate the VQ-vae layers and attach our forward hook to them
        self._hooks = {}
        self._names = {}
        self._register(module, cls)
        self._enabled = False

    def __enter__(self) -> object:
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._enabled = False

    def _register(self, module: nn.Module, cls: type) -> None:
        for nom, mod in module.named_modules():
            if isinstance(mod, cls) and mod not in self._hooks:
                self._hooks[mod] = mod.register_forward_hook(self._hook)
                self._names[mod] = nom

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: VQEOutput,
    ) -> None:
        # don't do anything OUTSIDE the with-scope
        if self._enabled:
            if module not in self._hooks:
                raise RuntimeError

            return self.on_forward(module, inputs, output)


# nt-s for more convenient access to the runtime states of the EMA updater
StateAcc = namedtuple("StateAcc", "size,vecs")
StateEMA = namedtuple("StateEMA", "size,vecs,alpha,eps")


class VQEMAUpdater(BaseVQHelper):
    """Update the embedding weights using exponential moving average centroids.

    Details
    -------
    The following is a replacement for the ordinary optimizer, that is
    specifically written for the VQ layers. It updates the embedding weights
    using the exponential moving average estimate of the cluster centroids.
    The intermediate stats for $k$ -means clustering are collected on each
    forward pass through the vq-layers this object is attached to. The EMA
    stats of the centroids and the embedding dictionary are updated only when
    the `.step` method is explicitly called.

    ToDo
    ----
    * allow customizable per-layer alpha settings, like the generic Optimizer.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        # embedding loss update speed
        alpha: float,
        # Laplace correction coefficient
        eps: float = 1e-3,
        # whether to actually update the embedding or just track ema stats
        update: bool = True,
    ) -> None:
        if not (0 < alpha <= 1):
            raise ValueError(f"`alpha` must be in (0, 1]. Got `{alpha}`.")
        super().__init__(module)

        # `*_acc` tensors accumulate clustering statistics between `.step`
        #  calls, and essentially play the role of tensor's `.grad` w.r.t
        #  optimizer's `.step`
        # `*_ema` buffers accumulate the historical clustering stats using
        #  exponential moving average with the specified `alpha` decay.
        self._acc, self._ema = {}, {}
        self.alpha, self.eps, self._update = alpha, eps, update

        # prevent the autograd from recording operations on the embeddings,
        #  only in case when updates are enabled
        if update:
            for mod in self._hooks:
                mod.weight.requires_grad_(False)

    def on_forward(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: VQEOutput,
    ) -> None:
        """Compute the cluster centroids for the given input and affinity."""
        # make sure our assumptions are correct
        input, *remaining = inputs
        assert not remaining

        # if `input` is a numeric tensor, then it has been clustered by
        #  the layer. Therefore its output is usseable for updating the
        #  parameters of the VQ clustering. Otherwise quit (non-diffable
        #  affinity data).
        if not input.is_floating_point():
            return

        # we use EMA only if the embedding parameter does not require grad
        if self._update and module.weight.requires_grad:
            raise RuntimeError(f"The VQ-VAE `{module}` has grads enabled.")

        # Compute the assignments and unnormalized barycentres
        # XXX $\mu_j = \frac{S_j}{n_j}$ the cluster centroid like in k-means
        #     `size[j]` is $n_j = \lvert i: k_i = j \rvert$
        #     `vecs[j]` is $S_j = \sum_i 1_{k_i = j} x_i$
        affinity = F.one_hot(output.indices, module.num_embeddings).to(input)
        data = input.detach().movedim(module.dim, -1)
        vecs = torch.einsum("...f, ...k -> kf", data, affinity)
        size = torch.einsum("...k -> k", affinity)
        # XXX `x.movedim(j, -1)` does `x.permute(*dims[:j], dims[j+1:], j)`

        # update the accumulators, but first lazily init them
        if self._acc.get(module) is None:
            self._acc[module] = StateAcc(
                torch.zeros_like(size),
                torch.zeros_like(vecs),
            )

        acc = self._acc[module]
        acc.size.add_(size)
        acc.vecs.add_(vecs)

    def step(self) -> None:
        """Update the exponential moving averages with the centroid data
        and then the embedding vectors.
        """
        # perform the EMA updates
        for mod in self._hooks:
            # if `_acc` for this module has not been initialized,
            #  ignore it. Otherwise, get the data and immediately
            #  invalidate it.
            acc = self._acc.pop(mod, None)
            if acc is None:
                continue

            # as for the `_ema` data, lazily init it
            ema = self._ema.get(mod)
            if ema is None:
                weight = mod.weight.detach()
                ema = self._ema[mod] = StateEMA(
                    torch.ones_like(weight[:, 0]),
                    weight.clone(),
                    self.alpha,
                    self.eps,
                )

            # XXX we could implement centroid repulsion on \hat{\mu} stored in `acc`
            pass

            # Commit the accumulated cluster sizes and unnormalized centroids
            #  into the EMA buffers
            # XXX `torch.lerp(a, b, w) = a.lerp(b, w) = (1 - w) * a + w * b`
            ema.size.lerp_(acc.size, ema.alpha)
            ema.vecs.lerp_(acc.vecs, ema.alpha)

            # flush the accumulators (non necessary since we `.pop`)
            acc.size.zero_()
            acc.vecs.zero_()

            if not self._update:
                continue

            # Apply the \epsilon-Laplace correction to new cluster sizes and
            #  update the embeddings inplace with the new centroids
            n = float(ema.size.sum())
            coef = n / (n + len(ema.size) * ema.eps)
            size = coef * (ema.size + ema.eps).unsqueeze(1)  # C x 1
            mod.weight.data.copy_(ema.vecs.div(size))  # C x F

    @property
    def entropy(self):
        out = {}
        for mod, ema in self._ema.items():
            if ema is None:
                continue

            # compute the entropy
            prob = ema.size / float(ema.size.sum())
            entropy = -F.kl_div(prob.new_zeros(()), prob, reduction="sum")
            out[mod] = float(entropy) / math.log(2)

        return out


class VQLossHelper(BaseVQHelper):
    r"""A context object, which tracks the embed-commit losses of the VQ layers
    in the module.

    Details
    -------
    This scope-object attaches itself to every VQ-vae layer in the specifie
    module and collects the commitment and embedding losses on each forward
    pass through the tracked modules.

    The losses are from van den Oord et al. (2017; p. 4 eq. 3.)
    $$
        \mathcal{L}
            = - \log p(x \mid q(x))   % reconstruction term
            + \|sg(z(x)) - q(x)\|^2   % embedding loss (dictionary update)
            + \|z(x) - sg(q(x))\|^2   % encoder's commitment loss
        \,, $$
    where $z(x)$ (`input`) is the data to be vector-quantized, e.g. coming
    from an upstream encoder network, $q(x) = e_{k(x)}$ with $
        k(x) = \arg\min_k \|z(x) - e_k\|^2
    $ are the resulting quantized vectors. `sg` is the stop-grad operator.

    The embeddings $e_\cdot$ receive no feedback from the reconstruction
    term (Ibid; p. 4 par. 3) due to the straight-through gradient estimator
    (implemented in the VQ layer itself):
    $$
        q(x) = x + sg\bigl( e_{k(x)} - x \bigr)
        \,. $$
    Instead the embeddings are learnt using either by an updated similar to
    $k$-means clustering, or by descending on the $\ell_2$-embeding loss.

    For each embedding the commitment loss is equivalent to a sum of two
    terms: the sample variance of the data vectors that are represented by
    it and the squared $\ell_2$ proximity of the non-diffable embedding to
    the sample mean of the data, which is the centroid of the isotropic
    $k-means. This decomposition implies that this term encourages tightly
    clustered input data and more precisely quantized vectors,
    (ibid, p.4 par. 4). Additionally, the loss aims to reduce the bias of
    the straight-through grad estimator.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        # diffable commitment loss weight
        beta: float = 0.25,
        # loss reduction method (sum or mean)
        reduction: str = "sum",
    ) -> None:
        assert reduction in ("sum", "mean")
        super().__init__(module)
        self.reduction = reduction
        self.beta = beta
        self._losses = {}

    def on_forward(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor],
        output: VQEOutput,
    ) -> None:
        r"""Compute the commitment and embedding losses over the forward pass data."""

        # make sure our assumptions are correct
        input, *remaining = inputs
        assert not remaining

        # compute the losses only if the input is numeric
        if not input.is_floating_point():
            return

        # Save the losses in lists in case the same layer is passed more
        #  than once within the active collection scope.
        self._losses.setdefault(module, []).append(
            (
                # the embedding loss is computed regardless of whether the module
                #  is in EMA mode or not, although it is never needed when in EMA
                F.mse_loss(output.vectors, input.detach(), reduction=self.reduction),
                # the commitment loss controls the within-cluster sum of squares
                #  and is useful only if the `input` is diffable numeric data
                F.mse_loss(input, output.vectors.detach(), reduction=self.reduction),
            )
        )

    def finish(self) -> dict[nn.Module, torch.Tensor]:
        """Finalise the collection and return a dict of losses."""

        # we average across multiple passes through the same layer
        # XXX the losses could potentially be non-diffable
        losses = {}
        for mod, ell in self._losses.items():
            emb, com = map(sum, zip(*ell))
            losses[mod] = (emb + self.beta * com) / len(ell)

        self._losses.clear()

        return losses
