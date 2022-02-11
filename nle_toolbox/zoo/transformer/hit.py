import torch
from torch import nn

from einops import repeat

from .vit import TransformerLayer, pair


class HiT(nn.Module):
    """Highway input-output transformer.
        * i-o port embeddings
    """
    def __init__(
        self,
        size,
        /,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        n_cls: int = 4,
        n_io: int = 8,
        head_size: int = None,
        dropout: float = 0.,
        *,
        n_layers: int = 1,
    ):
        super().__init__()

        n_rows, n_cols = pair(size)

        self.posemb = nn.Parameter(torch.randn(
            n_rows * n_cols + 3 * n_io + n_cls,
            embedding_dim
        ))

        self.cl = nn.Parameter(torch.randn(
            1, n_cls, embedding_dim,
        ))

        self.io = nn.Parameter(torch.randn(
            1, n_io, embedding_dim,
        ))

        self.ct = nn.Parameter(torch.randn(
            1, n_io, embedding_dim,
        ))

        self.layers = nn.ModuleList([
            TransformerLayer(
                embedding_dim,
                num_attention_heads,
                intermediate_size,
                head_size,
                dropout,
            ) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)
        self.n_cls, self.n_io = n_cls, n_io

    def forward(self, input, hx=None):
        out = []

        # input is T x B x N x F
        n_seq, n_batch, n_len = input.shape[:3]
        if hx is None:
            hx = repeat(
                torch.zeros_like(self.io),
                '() N F -> B N F', B=n_batch
            )

        for x in input:
            # cls tokens, i/o out, i/o inp, input
            x = torch.cat([
                repeat(self.cl, '() N F -> B N F', B=n_batch),
                repeat(self.io, '() N F -> B N F', B=n_batch),
                repeat(self.ct, '() N F -> B N F', B=n_batch),
                hx,   # B x N x F
                x,    # B x S x F
            ], dim=1).add(self.posemb)

            for layer in self.layers:
                x, _ = layer(x)

            x = self.norm(x)

            # extract the cls, i/o out and ctr embeddings
            x_cls, x_ioo, x_ctr, _, _ = torch.split(x, [
                self.n_cls, self.n_io, self.n_io, self.n_io, n_len,
            ], dim=1)
            hx = hx.lerp(x_ioo, x_ctr.sigmoid())

            out.append(x_cls.reshape(n_batch, -1))

        return torch.stack(out, dim=0), hx
