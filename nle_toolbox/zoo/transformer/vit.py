"""
This implementation of ViT is adapted almost verbatim from

    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
import math
import torch

from torch import nn
from torch import Tensor

from einops import repeat, rearrange


def pair(x, n=2):
    return x if isinstance(x, tuple) else (x,) * n


def expand_mask(mask, n_outputs, *, tok_to_out=False, out_to_all=True):
    shape = dict(zip("BNM", mask.shape))

    # XXX we need to expand the mask if has been provided
    B, N = shape["B"], shape["N"]

    # mask is B N M boolean tensor. we forbid outout-output attn, allow
    #  outputs to attendt to tokens, and optionally permit tokens to attend
    #  to outputs
    blk1 = mask.new_ones((n_outputs, n_outputs + N))
    if out_to_all:
        torch.eye(n_outputs, out=blk1[:, :n_outputs])
    blk1.logical_not_()

    blk2 = mask.new_full((N, n_outputs), not tok_to_out)
    return torch.cat(
        [
            repeat(blk1, "N M -> B N M", B=B),
            torch.cat([repeat(blk2, "N M -> B N M", B=B), mask], dim=-1),
        ],
        dim=-2,
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        *,
        head_size: int = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        if head_size is None:
            head_size, rem = divmod(embedding_dim, num_attention_heads)
            if rem > 0:
                raise ValueError(
                    f"{embedding_dim} is not a multiple" f" of {num_attention_heads}."
                )

        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.num_attention_heads = num_attention_heads

        # (Q)uery (K)ey (V)alue projections
        self.qkv = nn.Linear(
            embedding_dim,
            3 * num_attention_heads * head_size,
            bias=False,
        )

        # re-projection with noise
        self.prj = nn.Linear(
            num_attention_heads * head_size,
            embedding_dim,
            bias=bias,
        )
        self.drp = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.head_size)
        self.qkv.weight.data.uniform_(-stdv, stdv)

        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.prj.weight.data.uniform_(-stdv, stdv)
        if self.prj.bias is not None:
            self.prj.bias.data.zero_()

    def forward(self, x: Tensor, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        # `x` is `B N C`, `S` is `stack x 3`, and H -- # of heads
        # XXX in non-self attention the query sequence might have different
        #  size, in which case we would have to make a separate layer for Q,
        #  e.g. x is `B N C`, q is `B M C`
        que, key, val = rearrange(
            self.qkv(x),
            "B N (S H D) -> S B H N D",
            S=3,
            H=self.num_attention_heads,
        )

        # scaled attention
        #  $a_{j t s} = \frac{q_{j t}^\top k_{j t s}}{\sqrt{d}}$
        #  $\alpha_{j t s} = \softmax(a_{j t s} + (- \infty) m_{j t s})_{s=1}^n$
        #  $y_{j t} = \sum_s \alpha_{j t s} v_{j t s}$
        # XXX `attn @ val -> out` gives [B H M N] @ [B H N D] -> [B H M D]
        dots = torch.matmul(que, key.transpose(-1, -2))
        if mask is not None:
            dots = dots.masked_fill(
                repeat(mask.to(bool), "B M N -> B H M N", H=1),
                -math.inf,
            )

        # attend, average and dimshuffle
        attn = torch.softmax(dots.div(math.sqrt(self.head_size)), dim=-1)
        mhsa = rearrange(attn.matmul(val), "B H M D -> B M (H D)")

        # reproject and dropout
        return self.drp(self.prj(mhsa)), attn


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        *,
        layernorm: nn.Module = nn.LayerNorm,
        gelu: nn.Module = nn.GELU,
        elementwise_affine: bool = True,
        head_size: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # we use pre-norm MH self-A, but optionally disable the learnable affine
        # transformation in the normalizer before the MHA, since it makes sense
        # to think about the compatibilities in the attention matrix (pre-softmax)
        # as _semantic_ covariances. Hence we would want the normalizer NOT to
        # translate the input hiddens to an arbitrary location (scaling if ok,
        # since it actually means a learnable temperature).
        self.pn1 = layernorm(
            embedding_dim,
            elementwise_affine=elementwise_affine,
        )
        self.mha = MultiHeadSelfAttention(
            embedding_dim,
            num_attention_heads,
            head_size=head_size,
            bias=True,
            dropout=dropout,
        )

        self.pn2 = layernorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            gelu(),
            nn.Linear(intermediate_size, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        y, attn = self.mha(self.pn1(x), mask)
        x = x + y
        output = x + self.mlp(self.pn2(x))

        return output, attn


class TransformerStack(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        n_layers: int = 1,
        *,
        elementwise_affine: bool = False,
        head_size: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embedding_dim,
                    num_attention_heads,
                    intermediate_size,
                    elementwise_affine=elementwise_affine,
                    head_size=head_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, list[Tensor]]:

        # `x` is B N C and transformer layers keep it that way, `a` is `B H N N`
        attn = []
        for layer in self.layers:
            x, a = layer(x)
            attn.append(a)

        return x, attn


class ViTEncoder(nn.Module):
    def __init__(
        self,
        size,
        /,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        n_layers: int = 1,
        *,
        b_mean: bool = False,
        elementwise_affine: bool = True,
        head_size: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        n_rows, n_cols = pair(size)
        self.posemb = nn.Parameter(
            torch.randn(
                1,
                n_rows * n_cols,
                embedding_dim,
            )
        )
        self.cls = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.stack = TransformerStack(
            embedding_dim,
            num_attention_heads,
            intermediate_size,
            n_layers,
            elementwise_affine=elementwise_affine,
            head_size=head_size,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.b_mean = b_mean

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        # flatten  spatial dims, dimshuffle, and add learnt positional embeddings
        x = rearrange(x, "B C H W -> B (H W) C").add(self.posemb)

        # prepend the CLS-token
        x = torch.cat((repeat(self.cls, "() N C -> B N C", B=len(x)), x), dim=1)

        # transformer returns a 'B (H W) C' tensor in `x`, attn is a list
        x, attn = self.stack(x, mask=None)  # XXX mask dose not know about CLS
        x = self.norm(x)

        # the hidden state corresponding to the first CLS token
        return x.mean(dim=1) if self.b_mean else x[:, 0], attn
