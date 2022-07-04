import torch
from torch import nn

from einops import repeat

from .vit import TransformerStack


class HiT(nn.Module):
    """Highway input-output transformer.
    * i-o port embeddings

    Inspireed by [Training Very Deep Networks](https://arxiv.org/abs/1507.06228)
    """

    def __init__(
        self,
        n_context,
        /,
        embedding_dim: int,
        num_attention_heads: int,
        intermediate_size: int,
        n_cls: int = 4,
        n_io: int = 8,
        head_size: int = None,
        dropout: float = 0.0,
        *,
        n_layers: int = 1,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.posemb = nn.Parameter(
            torch.randn(n_cls + n_io + n_io + n_context, embedding_dim)
        )

        self.cls = nn.Parameter(
            torch.randn(
                1,
                n_cls,
                embedding_dim,
            )
        )

        self.iox = nn.Parameter(
            torch.randn(
                1,
                1,
                n_io,
                embedding_dim,
            )
        )

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
        self.n_cls, self.n_io = n_cls, n_io

        self.hx_shape = self.iox.shape

    def forward(self, input, hx=None):
        out = []

        # input is T x B x N x F
        n_seq, n_batch, n_len = input.shape[:3]
        if hx is None:
            hx = repeat(torch.zeros_like(self.iox), "L () N F -> L B N F", B=n_batch)

        # remove the fake leading unit dim
        hx = hx.squeeze(0)
        for x in input:
            # cls tokens, ctrl i/o, i/o inp, input
            x = torch.cat(
                [
                    repeat(self.cls, "() K F -> B K F", B=n_batch),
                    repeat(self.iox.squeeze(0), "() N F -> B N F", B=n_batch),
                    hx,  # L x B x N x F
                    x,  # B x S x F
                ],
                dim=1,
            ).add(self.posemb)

            x, attn = self.stack(x)
            x = self.norm(x)

            # extract the cls, i/o and updated hx embeddings
            x_cls, x_iox, upd_hx, _ = torch.split(
                x,
                [
                    self.n_cls,
                    self.n_io,
                    self.n_io,
                    n_len,
                ],
                dim=1,
            )
            hx = hx.lerp(upd_hx.tanh(), x_iox.sigmoid())

            out.append(x_cls.reshape(n_batch, -1))

        return torch.stack(out, dim=0), hx.unsqueeze(0)
