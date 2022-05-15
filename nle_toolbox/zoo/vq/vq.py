import torch
from torch import nn
from torch import Tensor, LongTensor

from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter

from collections import namedtuple

VQEOutput = namedtuple("VQEOutput", "values,indices,vectors")
VQEOutput.__doc__ += """

Attributes
----------
values: torch.Tensor
    The real-valued vector-quantized embeddings (vectors from some finite
    collection) that properly pass gradient feedback to the inputs vectors
    (from the full real-valued continuous vector space).

indices: torch.Tensor
    The integer-valued codes that identify the embedding and the Voronoi cell
    to which the input was assigned. These are obtained as a byproduct of
    the vector-quantized encoding process.

vectors: torch.Tensor
    Technical field, needed for computing the embedding and commitment losses.

See Also
--------
The docs in `VectorQuantizedVAE` provide more details, discussion and intuition
regarding the vector-quantized encoding as a process.
"""


class VectorQuantizedVAE(nn.Module):
    r"""Vector-quantized VAE layer.

    Parameters
    ----------
    num_embeddings: int
        Size of the dictionary of embeddings.

    embedding_dim: int
        The size of each embedding vector.

    Terminology
    -----------
    It is important to note that the term `vector-quantized encoder` relates
    to a procedure that constrains the input vector as a WHOLE from the full
    real-valued continuous vector space to a finite discrete collection of
    learnt real-valued vectors. This should not be confused with quantizing
    individual components of a vector from reals to some finite set of values
    with lower bit-widths, e.g. fp32 to int8 fixed point arithmetic, or based
    on value histograms.

    Details
    -------
    The Vector-Quantized VAE of

        [van den Oord et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html)

    trains the embeddings for the nearest-neighbour-based vector-quantization
    and proposes a way to efficiently backprop gradients through them

    $$
    \operatorname{vq}(z; \mu)
        = \sum_k \mu_k 1_{R_k}(z)
        \,,
        \partial_z \operatorname{vq}(z; \mu) = \operatorname{id}
    \,. $$

    This vector-quantization operation corresponds to a degenerate conditional
    categorical rv $k^\ast_z$ with distribution $
        p(k^\ast_z = j\mid z)
            = 1_{R_j}(z)
    $ where

    $$
    R_j
        = \bigl\{
            z \colon \|z - \mu_j\|_2 < \min_{k \neq j} \|z - \mu_k\|_2
        \bigr\}
    \,, $$

    are the cluster affinity regions w.r.t. $\|\cdot \|_2$ norm (Voronoi
    partition). Note that to save on the runtime compute we can compute

    $$
    k^\ast_z
        = \arg \min_k
            \frac12 \| \mu_k \|_2^2 - \langle z, \mu_k \rangle
        \,. $$

    The authors propose STE for grads and mutual consistency losses for
    the embeddings:
    * $
        \| \operatorname{sg}(z) - \mu_{k^\ast_z} \|_2^2
    $ -- forces the embeddings to match the latent cluster's centroid (recall
    the $k$-means algo) **NB** in the paper they use just $\mu$, but in their
    latest code they use the selected embeddings, which makes more sense, when
    we view VQ as hidden-state k-means clustering
    * $
        \| z - \operatorname{sg}(\mu_{k^\ast_z}) \|_2^2
    $ -- forces the input vectors to be tight within the cluster they are
    assigned to and makes the straight-through estimator less biased (the
    commitment loss).

    Vector-quantization closely resembles a single EM-step of the Gaussian
    Mixture Model with improper components. Indeed, the K-means algorithm can
    be viewed as the Expectation-Maximization algorithm applied to a mixture
    $$
        p(x) = \sum_{j=1}^K \pi_j \mathcal{N}_d\bigl(
            x \big \vert \mu_j, \Sigma_j
        \bigr)
    \,, $$
    with equiweighted components $\pi_j = \frac1K$, isotropic covariance $
        \Sigma_j = \frac1\nu I
    $ with precision parameter $\nu \to 0$. In this limit the M-step computes $
      \mu_j = \frac{\sum_i z_i q_{ij}}{\sum_i q_{ij}}
    $ where $q_{ij} \in \Delta_K$ are the assignments. However, the E-step's
    soft cluster assignments, which are based on the Bayesian factors,
    degenerate into one-hot $\arg\max$ affinity. Given this observation, it is
    possible to view VQ VAE with embeddings trained via the Exponential Moving
    Averaging as a special case of a Gaussian mixture model with the M-step's
    results being regularized by the distance from the previous parameters.

    As such vector-quantization encoder may be seen as an unsupervised input
    denoiser through online K-means clustering with clever tricks to properly
    pass gradients via the straight through estimator and to ensure inputs
    that cluster more tightly via the commitment loss.
    """
    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    @torch.no_grad()
    def lookup(self, input: Tensor) -> LongTensor:
        """Lookup the index of the nearest embedding."""

        # k(z) = \arg \min_k \|E_k - z\|^2
        #      = \arg \min_k \|E_k\|^2 - 2 E_k^\top z + \|z\|^2
        # XXX no need to compute the norm fully since we cannot backprop
        #     through the cluster affinities.

        emb = self.weight
        sqr = (emb * emb).sum(dim=1)
        cov = torch.einsum("...j, kj -> ...k", input, emb)
        return torch.argmin(sqr.sub(cov, alpha=2), dim=-1)

    def fetch(self, indices: LongTensor, at: int = -1) -> Tensor:
        """Fetch embeddings and put their dim at position `at`."""

        dims = list(range(indices.ndim))

        # the integers in `indices` corresponds to the rows of the embedding
        #  matrix (dictionary), which represents cluster centroids
        vectors = F.embedding(indices, self.weight)

        # permute the dimensions so that the features sit at the correct dim
        at = (vectors.ndim + at) if at < 0 else at
        return vectors.permute(*dims[:at], indices.ndim, *dims[at:])

    def forward(self, input: Tensor) -> VQEOutput:
        # the `input` is a tensor of integer codes: there is no embedding and
        #  commitment losses since there are no input vectors to quantize
        # XXX we still provide all the data necessary for the losses, which
        #  would be non-diffable zero constants in this particular case.
        if not input.is_floating_point():
            vectors = self.fetch(input).detach()
            return VQEOutput(vectors, input, vectors)

        # Now, if `input` is a numeric tensor, this means that we can actually
        #  vector-quantize it. Thus we get its integer codes and embeddings
        indices = self.lookup(input)
        vectors = self.fetch(indices)

        # if the `input` is a non-diffable numeric tensor, then there is no use
        # in the commitment loss. However the embedding dictionary can still be
        #  updated either through the diffable embedding loss, or via the EMA
        #  tracking of the M-step's cluster centroids.
        if not input.requires_grad:
            return VQEOutput(vectors.detach(), indices, vectors)

        # Finally, a diffable numeric `input`, means that it can accept STE and
        #  commitment grads
        return VQEOutput(
            (vectors - input).detach_() + input,  # straight-through grad
            indices,  # for EMA stats
            vectors,  # for the embedding and commitment losses
        )

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class VQVAEEmbeddings(nn.Identity):
    """Extract real-valued vector-quantized embeddings from VQ outputs."""

    def __init__(self, module: VectorQuantizedVAE) -> None:
        if not isinstance(module, VectorQuantizedVAE):
            raise TypeError(
                f"{type(self).__name__} wraps VQ" f" layers directly. Got `{module}`."
            )
        super().__init__()
        self.wrapped = module

    def forward(self, input: Tensor) -> Tensor:
        # `out.values` are the diffable embeddings
        return self.wrapped(input).values


class VQVAEIntegerCodes(nn.Identity):
    """Extract integer-valued codes from VQ outputs."""

    def __init__(self, module: VectorQuantizedVAE) -> None:
        if not isinstance(module, VectorQuantizedVAE):
            raise TypeError(
                f"{type(self).__name__} wraps VQ" f" layers directly. Got `{module}`."
            )
        super().__init__()
        self.wrapped = module

    def forward(self, input: Tensor) -> Tensor:
        # `out.indices` are the cluster ids
        return self.wrapped(input).indices
