import math
from warnings import warn

from collections import defaultdict, namedtuple

import torch
from torch import nn
from torch.nn import functional as F


VQEOutput = namedtuple('VQEOutput', 'values,indices,vectors')
VQELoss = namedtuple('VQELoss', 'embedding,commitment,entropy')


def entropy(codes, n_symbols):
    """The empirical binary entropy of the given categorical sample.
    """
    # compute the entropy of the generated codes
    prob = torch.bincount(codes.flatten(), minlength=n_symbols) / codes.numel()
    value = -F.kl_div(prob.new_zeros(()), prob, reduction='sum')
    return float(value) / math.log(2)


class VQEmbedding(nn.Embedding):
    r"""Vector-quantized VAE embedding layer.

    Parameters
    ----------
    num_embeddings: int
        Size of the dictionary of embeddings.

    embedding_dim: int
        The size of each embedding vector.

    alpha: float
        The speed with which the dictionary of embeddings tracks the input
        vectors using the exponential moving average. Zero disables tracking
        and transforms the embedding vectors (aka cluster centroids) into
        learnable parameter.

    eps: float, default=1e-5
        The Laplacian correction coefficient for the moving average updates.

    update : str, default=None
        The pass on which the EMA updates to the embeddings are applied.
        Applicable only when `alpha` > 0 and MUST be either 'backward' or
        'forward'.

    Note
    ----
    My own implementation taken from, reworked and improved
        [coding/vq-VAE.ipynb](https://github.com/ivannz/general-scribbles)

    (not comitted as of 2022-03-02).

    Details
    -------
    The Vector-Quantized VAE of
        [van den Oord et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html)
    trains the nearest-neighbour-based quantization embeddings and proposes
    a way to backprop gradients through them:
    $$
    \operatorname{vq}(z; e)
        = \sum_k e_k 1_{R_k}(z)
        \,,
        \partial_z \operatorname{vq}(z; e) = \operatorname{id}
        \,. $$

    This corresponds to a degenerate conditional categorical rv
    $k^\ast_z$ with distribution $
        p(k^\ast_z = j\mid z)
            = 1_{R_j}(z)
    $ where
    $$
    R_j = \bigl\{
        z\colon
            \|z - e_j\|_2 < \min_{k\neq j} \|z - e_k\|_2
        \bigr\}
    \,, $$

    are the cluster affinity regions w.r.t. $\|\cdot \|_2$ norm. Note that we
    can compute

    $$
    k^\ast_z
        := \arg \min_k \frac12 \bigl\| z - e_k \bigr\|_2^2
        = \arg \min_k
            \frac12 \| e_k \|_2^2
            - \langle z, e_k \rangle
        \,. $$

    The authors propose STE for grads and mutual consistency losses for
    the embeddings:
    * $\| \operatorname{sg}(z) - e_{k^\ast_z} \|_2^2$ -- forces the embeddings
    to match the latent cluster's centroid (recall the $k$-means algo)
      * **NB** in the paper they use just $e$, but in the latest code they use
      the selected embeddings
      * maybe we should compute the cluster sizes and update to the proper
      centroid $
          e_j = \frac1{
              \lvert {i: k^\ast_{z_i} = j} \rvert
          } \sum_{i: k^\ast_{z_i} = j} z_i
      $.
    * $\| z - \operatorname{sg}(e_{k^\ast_z}) \|_2^2$ -- forces the encoder
    to produce the latents, which are consistent with the cluster they are
    assigned to (the commitment loss).

    (REWRTE THIS, focusing on the connection of K-means with GMM using EM algo)
    this is one step of the K-mean EM, with M being regularized by the distance
    from the previous improper GMM mixture.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.,
        *,
        update: str = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm=None,
            padding_idx=None,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.alpha, self.eps = alpha, eps

        # if `alpha` is zero then `.weight` is updated by means other than EMA
        self.register_buffer('ema_vecs', None)
        self.register_buffer('ema_size', None)
        if self.alpha <= 0:
            if update is not None:
                warn(
                    f"`update` = '{update}' has no "
                    "effect in NON moving average mode.",
                    RuntimeWarning,
                )

            return

        if not (0 < self.alpha <= 1):
            raise ValueError(f"`alpha` must be in (0, 1]. Got `{self.alpha}`.")

        if update not in ('forward', 'backward', 'manual'):
            raise ValueError(
                "In exponential moving average mode (`alpha` > 0) `update`"
                " must be either 'manual', 'forward' or 'backward'."
                f" Got `{update}`."
            )

        # demote `.weight` to a buffer and disable backprop for it
        # XXX can promote buffer to parameter, but not back, so we `delattr`.
        #  Also non-inplace `.detach` creates a copy NOT reflected in referrers
        weight = self.weight
        delattr(self, 'weight')
        self.register_buffer('weight', weight.detach_())

        # allocate buffer for tracking k-means cluster centroid updates
        self.register_buffer(
            'ema_vecs', self.weight.clone(),
        )
        self.register_buffer(
            'ema_size', torch.zeros_like(self.ema_vecs[:, 0]),
        )

        # update embeddings only in training mode on the specified pass
        def _update(module, *ignore):
            if module.training:
                module.update()

        if update == 'backward':
            self.register_full_backward_hook(_update)

        elif update == 'forward':
            self.register_forward_hook(_update)

    @torch.no_grad()
    def centroids(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the cluster centroids for the given data and affinity.
        """
        # `input` is `... x F` and `indices` are `...`
        affinity = F.one_hot(indices, self.num_embeddings).to(input)
        # XXX 'affinity' is `... x C`

        # sum the F-dim input vectors into bins by affinity
        #  `size[j]`:  n_j = \lvert i: k_i=j \rvert
        #  `vecs[j]`:  S_j = \sum_i 1_{k_i = j} x_i
        size = torch.einsum('...k -> k', affinity)
        vecs = torch.einsum('...f, ...k -> kf', input, affinity)

        return size, vecs  # XXX $\mu_j = \frac{S_j}{n_j}$ <<- centroid

    @torch.no_grad()
    def accumulate(
        self,
        size: torch.Tensor,
        vecs: torch.Tensor,
    ) -> None:
        """Update the exponential moving averages with the centroid data.
        """
        # raise if EMA updates were disabled (all `ema_*` buffers are None)
        assert self.alpha > 0

        # track cluster size and unnormalized vecs with EMA
        # XXX torch.lerp(a, b, w) = a.lerp(b, w) = (1 - w) * a + w * b
        self.ema_size.lerp_(size, self.alpha)
        self.ema_vecs.lerp_(vecs, self.alpha)

    @torch.no_grad()
    def update(self) -> None:
        """Update the embedding vectors from the accumulated EMA stats.
        """
        # Silently quit if EMA is disabled (all `ema_*` buffers are None)
        if self.alpha <= 0:
            return

        # Apply \epsilon-Laplace correction
        n = float(self.ema_size.sum())
        coef = n / (n + self.num_embeddings * self.eps)
        size = coef * (self.ema_size + self.eps).unsqueeze(1)
        self.weight.data.copy_(self.ema_vecs / size)

    @torch.no_grad()
    def lookup(self, input: torch.Tensor) -> torch.Tensor:
        """Lookup the index of the nearest embedding.
        """
        emb = self.weight
        # k(z) = \arg \min_k \|E_k - z\|^2
        #      = \arg \min_k \|E_k\|^2 - 2 E_k^\top z + \|z\|^2
        # XXX no need to compute the norm fully since we cannot backprop
        #  through the cluster affinities.

        sqr = (emb * emb).sum(dim=1)
        cov = torch.einsum('...j, kj -> ...k', input, emb)
        return torch.argmin(sqr - 2 * cov, dim=-1)

    def fetch(
        self,
        indices: torch.LongTensor,
        at: int = -1,
    ) -> torch.Tensor:
        """fetch embeddings and put their dim at position `at`"""
        vectors = super().forward(indices)  # call Embedding.forward

        # indices.shape is batch x *spatial
        dims = list(range(indices.ndim))
        at = (vectors.ndim + at) if at < 0 else at
        # vectors.permute(0, input.ndim-1, *range(1, input.ndim-1))
        return vectors.permute(*dims[:at], indices.ndim, *dims[at:])

    def forward(self, input: torch.Tensor) -> VQEOutput:
        """vq-VAE clustering with straight-through estimator.

        Details
        -------
        Implements

            [van den Oord et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html).

        See details in the class docstring.
        """

        # lookup the index of the nearest embedding and accumulate stats
        #  for moving average embedding updates on forward pass
        indices = self.lookup(input)  # E-step
        if self.training and self.alpha > 0:
            # slowly update embeddings with new cluster centroids
            self.accumulate(*self.centroids(input, indices))  # M-step

        # fetch the embedding vectors (cluster centroids)
        vectors = self.fetch(indices)

        # use the straight-through grad estimator: copy grad from q(x) to z(x)
        # XXX we return additional data for the losses (embedding and
        #  commitment) and diagnostics.
        return VQEOutput(input + (vectors - input).detach(), indices, vectors)

    def loss(
        self,
        input: torch.Tensor,
        output: VQEOutput,
        *,
        reduction: str = 'sum',
    ) -> tuple[torch.Tensor]:
        """Compute the commitment and embedding losses and the coding entropy.
        """
        assert isinstance(output, VQEOutput)

        # commitment and embedding losses from van den Oord et al. (2017; p. 4 eq. 3.)
        # loss = - \log p(x \mid q(x))
        #      + \|sg(z(x)) - q(x)\|^2   % embedding loss (dictionary update)
        #      + \|z(x) - sg(q(x))\|^2   % encoder's commitment loss
        # where z(x) is output of the encoder network (`input` variable)
        #       q(x) = e_{k(x)}, for k(x) = \arg\min_k \|z(x) - e_k\|^2
        # XXX `the embeddings receive no grad feedback from the reconstruction`
        embedding = F.mse_loss(
            output.vectors, input.detach(),
            reduction=reduction,
        )
        # XXX `embedding` loss is non-diffable if we use EMA updates

        # XXX p.4 `To make sure the encoder commits to an embedding and its
        #          output does not grow, since the volume of the embedding
        #          space is dimensionless.`
        # XXX we can compute the commitment loss using `output.detach()`
        commitment = F.mse_loss(
            input, output.vectors.detach(),
            reduction=reduction,
        )
        # XXX this reduces the bias of the straight-through grad estimator

        # compute the binary entropy of the produced codes
        ent = entropy(output.indices, self.num_embeddings)
        return VQELoss(embedding, commitment, ent)


class VQEmbeddingHelper:
    """A helper for seamless operation of VQ embedding layers.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        reduction: str = 'sum',
    ) -> None:
        assert reduction in ('mean', 'sum', 'none')

        self.hooks, self.names = {}, {}
        self.register(module)

        self.collected = defaultdict(list)
        self.reduction = reduction

    def clear(self) -> None:
        self.collected.clear()

    def register(self, module: nn.Module) -> None:
        """Attach output hooks to every VQ-vae layer in the module.
        """
        for nom, mod in module.named_modules():
            if not isinstance(mod, VQEmbedding):
                continue

            if mod not in self.hooks:
                self.names[mod] = nom
                self.hooks[mod] = mod.register_forward_hook(self._output_hook)

    def remove(self) -> None:
        self.clear()
        self.names.clear()

        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def _output_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: VQEOutput,
    ) -> torch.Tensor:
        if not isinstance(output, VQEOutput):
            return output

        self.collected[module].append(
            module.loss(*inputs, output, reduction=self.reduction)
        )

        return output.values

    def __iter__(self) -> None:
        if self.reduction == 'sum':
            fn = sum

        elif self.reduction == 'mean':
            fn = lambda x: sum(x) / len(x)

        elif self.reduction == 'none':
            fn = list

        for mod, dat in self.collected.items():
            emb, com, ent = zip(*dat)
            yield self.names[mod], (fn(emb), fn(com), sum(ent) / len(dat))

        self.clear()


class VQSendRecv(nn.Module):
    """Sender-receiver pair with Vector-quantized VAE layer.

    For experiments realted to the messaging mechanism proposed by
        [Havrilov and Titov (2017)](https://proceedings.neurips.cc/paper/2017/hash/70222949cc0db89ab32c9969754d4758-Abstract.html)

    also useful for implementing vq-vae MuZero
        [Ozair et al. (2021)](https://proceedings.mlr.press/v139/ozair21a.html)
    """
    def __init__(
        self,
        send: nn.Module,
        recv: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.01,
        *,
        update: str = None
    ) -> None:
        super().__init__()
        self.send = send
        self.recv = recv
        self.vq = VQEmbedding(
            num_embeddings,
            embedding_dim,
            alpha,
            update=update,
        )

    def forward(
        self,
        input: torch.Tensor,
        reduction: str = 'sum',
    ) -> tuple[torch.Tensor]:
        # `send: X -> R^{M F}` transforms [... C] -->> [... M F]`
        z = self.send(input)

        # `vq: R^F -> R^F` quantizes (k-means) and reembeds w. centroids
        out = self.vq(z, reduction)
        assert isinstance(out, VQEOutput)

        # `recv: R^{M F} -> X`
        x = self.recv(out.values)
        return x, out.indices, self.vq.loss(z, out, reduction)
