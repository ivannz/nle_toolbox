import math

from typing import Iterator
import torch

from torch import nn
from torch.nn import functional as F


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
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.,
        eps: float = 1e-5,
        *,
        auto: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm=None,
            padding_idx=None,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.alpha, self.eps, self.auto = alpha, eps, auto

        # if `alpha` is zero then `.weight` is updated by other means
        self.register_buffer('ema_vecs', None)
        self.register_buffer('ema_size', None)
        if self.alpha <= 0:
            return

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

    @torch.no_grad()
    def accumulate(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Update the embedding vectors by Exponential Moving Average.
        """

        # `input` is `... x F`, `indices` are `...`
        affinity = F.one_hot(indices, self.num_embeddings).to(input)
        # XXX 'affinity' is `... x C`

        # sum the F-dim input vectors into bins by affinity
        #  S_j = \sum_i 1_{k_i = j} x_i
        #  n_j = \lvert i: k_i=j \rvert
        upd_vecs = torch.einsum('...f, ...k -> kf', input, affinity)
        upd_size = torch.einsum('...k -> k', affinity)

        # track cluster size and unnormalized vecs with EMA
        # XXX torch.lerp(a, b, w) = a.lerp(b, w) = (1 - w) * a + w * b
        self.ema_vecs.lerp_(upd_vecs, self.alpha)
        self.ema_size.lerp_(upd_size, self.alpha)

    @torch.no_grad()
    def update(self) -> None:
        """Update the embedding vectors from the accumulated EMA stats.
        """
        # do not update is EMA is disabled (all `ema_*` buffers are None)
        if self.alpha <= 0:
            return

        # Apply \epsilon-Laplace correction
        n = self.ema_size.sum()
        coef = n / (n + self.num_embeddings * self.eps)
        size = coef * (self.ema_size + self.eps).unsqueeze(1)
        self.weight.data.copy_(self.ema_vecs / size)

    @torch.no_grad()
    def lookup(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup the index of the nearest embedding.
        """
        emb = self.weight
        # k(z) = \arg \min_k \|E_k - z\|^2
        #      = \arg \min_k \|E_k\|^2 - 2 E_k^\top z + \|z\|^2
        # XXX no need to compute the norm fully since we do not
        #  backprop through the input when clustering.

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

    def forward(
        self,
        input: torch.Tensor,
        reduction: str = 'sum',
    ) -> tuple[torch.Tensor]:
        """vq-VAE clustering with straight-through estimator and commitment
        losses.

        Details
        -------
        Implements

            [van den Oord et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html).

        See details in the class docstring.
        """

        # lookup the index of the nearest embedding and fetch it
        indices = self.lookup(input)
        vectors = self.fetch(indices)

        # commitment and embedding losses from van den Oord et al. (2017; p. 4 eq. 3.)
        # loss = - \log p(x \mid q(x))
        #      + \|sg(z(x)) - q(x)\|^2   % embedding loss (dictionary update)
        #      + \|z(x) - sg(q(x))\|^2   % encoder's commitment loss
        # where z(x) is output of the encoder network
        #       q(x) = e_{k(x)}, for k(x) = \arg\min_k \|z(x) - e_k\|^2
        # XXX p.4 `To make sure the encoder commits to an embedding and its
        #          output does not grow, since the volume of the embedding
        #          space is dimensionless.`
        # XXX `the embeddings receive no grad feedback from the reconstruction`
        embedding = F.mse_loss(vectors, input.detach(), reduction=reduction)
        commitment = F.mse_loss(input, vectors.detach(), reduction=reduction)

        # the straight-through grad estimator: copy grad from q(x) to z(x)
        output = input + (vectors - input).detach()

        # accumulate the clusterting stats with the exponential moving average
        #  regardless of the mode we're in, but update only in training mode
        if self.alpha > 0:
            self.accumulate(input, indices)

            # update the weights only in training mode when auto-update is on
            # XXX `embedding` loss is non-diffable if we use ewm updates
            if self.training and self.auto:
                self.update()

        # compute the binary entropy
        ent = entropy(indices, self.num_embeddings)
        return output, indices, embedding, commitment, ent


def named_vq_modules(
    module: nn.Module,
    prefix: str = '',
) -> Iterator[tuple[str, nn.Module]]:
    """Yield all VQ-vae layers in the module.
    """

    for nom, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, VQEmbedding):
            yield nom, mod


def update_vq_modules(module: nn.Module) -> None:
    """Call `.update` on every child VQ layer in the module, regardless of
    their `.training` mode or `.auto` setting.
    """

    for _, mod in named_vq_modules(module):
        mod.update()


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
        auto: bool = True
    ) -> None:
        super().__init__()
        self.send = send
        self.recv = recv
        self.vq = VQEmbedding(
            num_embeddings,
            embedding_dim,
            alpha=alpha,
            auto=auto,
        )

    def forward(
        self,
        input: torch.Tensor,
        reduction: str = 'sum',
    ) -> tuple[torch.Tensor]:
        # `send: X -> R^{M F}` transforms [... C] -->> [... M F]`
        z = self.send(input)

        # `vq: R^F -> R^F` quantizes (k-means) and reembeds w. centroids
        emb, codes, embedding, commitment, entropy = self.vq(z, reduction)

        # `recv: R^{M F} -> X`
        x = self.recv(emb)
        return x, codes, embedding, commitment, entropy
