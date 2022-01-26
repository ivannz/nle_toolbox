import torch
import numpy as np

import torch.nn.functional as F

from typing import Optional
from nle.nethack import (
    MAX_GLYPH,
    # NO_GLYPH,
    # NLE_BLSTATS_SIZE,
    # NLE_INVENTORY_SIZE,
    NLE_BL_X,
    NLE_BL_Y,
)

from einops import rearrange

from ...utils.nn import bselect
from ...utils.env.defs import glyphlut
from ...utils.env.defs import MAX_ENTITY, glyph_group


class LegacyGlyphEmbedding(torch.nn.Module):
    """Glyph Embedding is the shared representation layer."""
    def __init__(self, embedding_dim=128):
        super().__init__()

        # glyph to group-entity lookup
        self.register_buffer(
            'gl_lookup', torch.tensor(np.c_[
                glyphlut.group,
                glyphlut.entity,
            ]).clone()
        )

        # glyph's entity encoder
        self.entity = torch.nn.Embedding(
            MAX_ENTITY + 1,
            int(4 * embedding_dim // 5),  # XXX why do we use this ratio?
            padding_idx=MAX_ENTITY,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

        # glyph's entity encoder
        self.group = torch.nn.Embedding(
            glyph_group.MAX + 1,
            embedding_dim - self.entity.embedding_dim,
            padding_idx=glyph_group.MAX,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

    def lookup(self, glyphs):
        """Look up the group and entity indices for the given glyphs."""
        return torch.unbind(self.gl_lookup[glyphs.long()], dim=-1)

    def forward(self, glyphs):
        """Embed the input glyphs."""
        grp, ent = self.lookup(glyphs)
        return torch.cat([self.entity(ent), self.group(grp)], dim=-1)


class GlyphEmbedding(torch.nn.Embedding):
    """Glyph-entity embedding layer is the shared representation layer.
    """
    from nle_toolbox.utils.env.defs import MAX_ENTITY

    def __init__(
        self,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        # glyph's entity encoder
        super().__init__(
            self.MAX_ENTITY + 1,
            embedding_dim,
            padding_idx=self.MAX_ENTITY,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

        # glyph-to-entity lookup
        self.register_buffer(
            'lookup', torch.tensor(glyphlut.entity).clone(),
        )

    def forward(self, input: torch.Tensor):
        """Look up the entities of the given glyphs and embed them.
        """
        return super().forward(self.lookup[input.long()])


class GlyphFeatures(torch.nn.Module):
    """This service layer performs glyph embedding form the inventory and
    the screen glyphs. From the latter it extracts a vicinity of the specified
    radius about the coordinates given in the bottom line stats.
    """
    def __init__(self, glyphs, window=1):
        super().__init__()
        self.glyphs = glyphs
        self.window = window

    def forward(self, obs):
        k = self.window

        # turn T x B x R x C into `T x B x (k+R+k) x (k+C+k) x ...`
        #  by pading symmetrically with invalid glyphs and embedding.
        glyphs = obs['glyphs']
        gl_padded = self.glyphs(F.pad(glyphs, (k,) * 4, value=MAX_GLYPH))
        # XXX we embed extra 2(R + C) + 4 glyphs, i.e. about 13% overhead

        # use stride trix to dimshuffle and unfold into sliding local windows
        #  of the form `T x B x R x C x ... x (k+1+k) x (k+1+k)`
        # XXX could we have used einops here? No, as it does not provide
        #  rearranging dims with overlappng strides.
        n_seq, n_batch, n_rows, n_cols = glyphs.shape
        n_features = gl_padded.shape[4:]
        s_seq, s_batch, s_rows, s_cols, *s_features = gl_padded.stride()
        gl_windows = gl_padded.as_strided(
            # fold R x C into 2d windows in dims after F
            (n_seq, n_batch, n_rows, n_cols, *n_features, k+1+k, k+1+k),

            # when mul-ing dims in shape use the lowest stride!
            (s_seq, s_batch, s_rows, s_cols, *s_features, s_rows, s_cols),
        )

        # extract vicinities around the row-col coordinates, specified in bls
        bls = obs['blstats']
        gl_vicinity = bselect(
            gl_windows,
            bls[..., NLE_BL_Y],
            bls[..., NLE_BL_X],
            dim=2,
        )

        # now permute the embedded glyphs to T x B x ... x R x C
        gl_screen = rearrange(gl_padded[:, :, k:-k, k:-k],
                              'T B R C ... -> T B ... R C')

        # embed inventory glyphs
        # XXX need to replace NO_GLYPH with MAX_GLYPH, unless they coincide.
        gl_inventory = rearrange(self.glyphs(obs['inv_glyphs']),
                                 'T B N ... -> T B ... N')

        # final touch is to extract the embedding of `self` on the map
        gl_self = gl_vicinity[..., k, k]
        return dict(
            screen=gl_screen.contiguous(),
            vicinity=gl_vicinity,  # already contiguous
            inventory=gl_inventory.contiguous(),
            self=gl_self.contiguous(),
        )


class LegacyGlyphFeatures(GlyphFeatures):
    def __init__(self, embedding_dim, window=1):
        super().__init__(LegacyGlyphEmbedding(embedding_dim), window)


class GlyphEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # the upstream embedding layer
        self.embedding = GlyphEmbedding(embedding_dim)

        # for dungeon glyphs we use a transformer layer
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=8,
                dim_feedforward=embedding_dim * 4,
            ),
            num_layers=1,
        )

    def forward(self, glyphs):
        # glyphs ... x 21 x 79
        dims = dict(zip('TBHW', glyphs.shape))

        # 1. embed the dungeon layout
        # XXX positional encodings might be necessary
        glyphs = self.embedding(glyphs)
        layout = rearrange(
            self.transformer(
                rearrange(glyphs, 'T B H W C -> (H W) (T B) C')),
            '(H W) (T B) C -> T B H W C', **dims)

        # 2. embed the surrounding glyphs

        return layout
