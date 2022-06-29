"""
This implementation of ViT is adapted almost verbatim from

    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
import math
import torch

from torch import nn
from collections import OrderedDict

from torch import Tensor
from einops import repeat, rearrange


def pair(x, n=2):
    return x if isinstance(x, tuple) else (x,) * n


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        head_size: int = None,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        return_attention: bool = False,
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
        self.return_attention = return_attention

        # (Q)uery (K)ey (V)alue projections
        self.qkv = nn.Linear(
            embedding_dim,
            3 * num_attention_heads * head_size,
            bias=False,
        )

        # re-projection with noise
        self.prj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "proj",
                        nn.Linear(
                            num_attention_heads * head_size,
                            embedding_dim,
                            bias=bias,
                        ),
                    ),
                    ("drop", nn.Dropout(dropout)),
                ]
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.head_size)
        self.qkv.weight.data.uniform_(-stdv, stdv)

        # (qkv) Kaiming-like uniform init based on head fan-out
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        self.prj.proj.weight.data.uniform_(-stdv, stdv)
        if self.prj.proj.bias is not None:
            self.prj.proj.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # qkv is x is `B N C`, below S is `stack x 3`, and H -- # of heads
        # XXX in non-self attention the query sequence might have different
        #  size, in which case we would have to make a separate layer for Q,
        #  e.g. x is `B N C` and q is `B M C`
        que, key, val = rearrange(
            self.qkv(x),
            "B N (S H D) -> S B H N D",
            S=3,
            H=self.num_attention_heads,
        )

        # scaled attention
        #  $a_{j t s} = \frac{q_{j t}^\top k_{j s}}{\sqrt{d}}$
        #  $\alpha_{j t s} = \softmax(a_{j t s} + (- \infty) m_{j t s})_{s=1}^n$
        #  $y_{j t} = \sum_s \alpha_{j t s} v_{j s}$
        # XXX `attn @ val -> out` gives [B H M N] @ [B H N D] -> [B H M D]
        dots = torch.matmul(que, key.transpose(-1, -2))
        if mask is not None:
            # in-place masking is diffable
            dots = dots.masked_fill(
                repeat(mask.to(bool), "B M N -> B H M N", H=1),
                -math.inf,
            )

        attn = torch.softmax(dots.div(math.sqrt(self.head_size)), dim=-1)
        out = rearrange(attn.matmul(val), "B H M D -> B M (H D)")

        # reproject and dimshuffle
        out = self.prj(out)
        return (out, attn) if self.return_attention else out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        head_size: int = None,
        dropout: float = 0.0,
        *,
        layernorm: nn.Module = nn.LayerNorm,
        gelu: nn.Module = nn.GELU,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        # we use pre-norm MH self-A, but optionally disable the learnable affine
        # transformation in the normalizer before the MHA, since it makes sense
        # to think about the compatibilities in the attention matrix (pre-softmax)
        # as _semantic_ covariances. Hence we would want the normalizer NOT to
        # translate the input hiddens to an arbitrary location (scaling if ok,
        # since it actually means a learnable temperature).
        self.attn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "norm",
                        layernorm(
                            embedding_dim,
                            elementwise_affine=elementwise_affine,
                        ),
                    ),
                    (
                        "attn",
                        MultiHeadAttention(
                            embedding_dim,
                            num_attention_heads,
                            head_size,
                            dropout,
                            return_attention=True,
                        ),
                    ),
                ]
            )
        )

        self.pwff = nn.Sequential(
            OrderedDict(
                [
                    ("norm", layernorm(embedding_dim)),
                    ("ff_1", nn.Linear(embedding_dim, intermediate_size)),
                    ("gelu", gelu()),
                    ("ff_2", nn.Linear(intermediate_size, embedding_dim)),
                    ("drop", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y, attn = self.attn(x)
        x = y + x
        x = self.pwff(x) + x
        return x, attn


class ViTEncoder(nn.Module):
    def __init__(
        self,
        size,
        /,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        head_size: int = None,
        dropout: float = 0.0,
        *,
        n_layers: int = 1,
        b_mean: bool = False,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        n_rows, n_cols = pair(size)

        self.posemb = nn.Parameter(
            torch.randn(
                1,
                n_rows * n_cols + 1,
                embedding_dim,
            )
        )
        self.cls = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embedding_dim,
                    num_attention_heads,
                    intermediate_size,
                    head_size,
                    dropout,
                    elementwise_affine=elementwise_affine,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.b_mean = b_mean

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # cls-token and positional embedding
        x = (
            torch.cat(
                (
                    repeat(self.cls, "() N C -> B N C", B=len(x)),
                    rearrange(x, "B C H W -> B (H W) C"),
                ),
                dim=1,
            )
            + self.posemb
        )

        # transformer keeps `x` as '(T B) (H W) C'
        attentions = []
        for layer in self.layers:
            x, attn = layer(x)
            attentions.append(attn)

        # each attn in `b h n n`
        attentions = torch.stack(attentions, dim=1)

        x = self.norm(x)

        # the hidden state corresponding to the first CLS token
        return x.mean(dim=1) if self.b_mean else x[:, 0], attentions
