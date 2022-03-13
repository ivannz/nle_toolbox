from nle.nethack import (
    MAX_GLYPH,
    # NO_GLYPH,
    # NLE_BLSTATS_SIZE,
    # NLE_INVENTORY_SIZE,
    NLE_BL_X,
    NLE_BL_Y,
)

import torch
from torch import nn

from einops import rearrange
from torch.nn import functional as F

from ..utils.nn import bselect
from ..utils.env.defs import glyphlut, glyph_group, MAX_ENTITY


class GlyphEmbedding(nn.Module):
    """A generic glyph embedder.
    """
    def __init__(self, embedding_dim):
        super().__init__()

        # glyph-to-entity embedding
        self.entity = nn.Embedding(
            MAX_ENTITY + 1, embedding_dim, padding_idx=MAX_ENTITY,
        )
        self.entity.register_buffer(
            'lut', torch.tensor(glyphlut.entity).clone(),
        )

        # glyph-to-group embedding
        self.group = nn.Embedding(
            glyph_group.MAX + 1, embedding_dim, padding_idx=glyph_group.MAX,
        )
        self.group.register_buffer(
            'lut', torch.tensor(glyphlut.group).clone(),
        )

    def forward(self, glyphs):
        # `glyphs` has shape `...`
        glyphs = glyphs.long()

        # glyph -->> W_e[glyph.entity] + W_g[glyph.group]
        ent = self.entity(self.entity.lut[glyphs])
        grp = self.group(self.group.lut[glyphs])
        return ent + grp


class EgoCentricEmbedding(nn.Module):
    """Learnable additive embedding to the centre of the ego-centric symmetric
    glpyh view into the map.
    """
    def __init__(self, embedding_dim):
        super().__init__()

        # ego embedding offset
        # TODO combine `NLE_BL_HD` (monster level) with ego-embedding?
        self.ego = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, vicinity):
        # vicinity is `... x (k + 1 + k) x (k + 1 + k) x F`
        Hq, Hr = divmod(vicinity.shape[-3], 2)
        Wq, Wr = divmod(vicinity.shape[-2], 2)
        assert Hr == Wr == 1  # symmetric window

        # add ego at the centre of the vicinity
        # XXX  `.pad` pads dims from tail
        padding = (0, 0, Hq, Hq + Hr - 1, Wq, Wq + Wr - 1)
        return vicinity.add(F.pad(self.ego, padding))


class GlyphFeatureExtractor(nn.Module):
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
