"""
This implementation of ViT is adapted almost verbatim from

    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
import math
import torch

from torch import nn
from collections import OrderedDict

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
    ):
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
        self.prj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "proj",
                        nn.Linear(
                            num_attention_heads * head_size,
                            embedding_dim,
                            bias=True,
                        ),
                    ),
                    ("drop", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x):
        # qkv is x is `B N C`, below S is `stack x 3`, and H -- # of heads
        que, key, val = rearrange(
            self.qkv(x),
            "B N (S H D) -> S B H N D",
            S=3,
            H=self.num_attention_heads,
        )

        # scaled attention
        #  $a_{j t s} = \frac{q_{j t}^\top k_{j s}}{\sqrt{d}}$
        #  $\alpha_{j t s} = \softmax(a_{j t s})_{s=1}^n$
        #  $y_{j t} = \sum_s \alpha_{j t s} v_{j s}$
        # XXX `attn @ val -> out` gives [B H N N] @ [B H N D] -> [B H N D]
        dots = torch.matmul(que, key.transpose(-1, -2))
        attn = torch.softmax(dots.div(math.sqrt(self.head_size)), dim=-1)
        out = rearrange(attn.matmul(val), "B H N D -> B N (H D)")

        # reproject and dimshuffle
        return self.prj(out), attn


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        head_size: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn = nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.LayerNorm(embedding_dim)),
                    (
                        "attn",
                        MultiHeadAttention(
                            embedding_dim,
                            num_attention_heads,
                            head_size,
                            dropout,
                        ),
                    ),
                ]
            )
        )

        self.pwff = nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.LayerNorm(embedding_dim)),
                    ("ff_1", nn.Linear(embedding_dim, intermediate_size)),
                    ("gelu", nn.GELU()),
                    ("ff_2", nn.Linear(intermediate_size, embedding_dim)),
                    ("drop", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x):
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
    ):
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
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.b_mean = b_mean

    def forward(self, x):
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
