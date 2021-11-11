import torch
import torch.nn.functional as F

from nle.nethack import (
    MAX_GLYPH,
    # NO_GLYPH,
    # NLE_BLSTATS_SIZE,
    # NLE_INVENTORY_SIZE,
    NLE_BL_X,
    NLE_BL_Y,
)

from einops import rearrange

from ...utils.env.defs import glyphlut
from ...utils.env.defs import MAX_ENTITY, glyph_group


class GlyphEmbedding(torch.nn.Module):
    """Glyph Embedding is the shared representation layer."""
    def __init__(self, embedding_dim=128):
        super().__init__()

        # glyph to group-entity lookup
        self.register_buffer(
            'gl_lookup', torch.tensor([
                glyphlut.group,
                glyphlut.entity,
            ]).T.clone()
        )

        # glyph's entity encoder
        self.entity = torch.nn.Embedding(
            MAX_ENTITY + 1,
            int(4 * embedding_dim // 5),
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

    def forward(self, glyphs):
        grp, ent = torch.unbind(self.gl_lookup[glyphs.long()], dim=-1)
        return torch.cat([self.entity(ent), self.group(grp)], dim=-1)


class GlyphFeatures(torch.nn.Module):
    """This service layer performs glyph embedding form the inventory and
    the screen glyphs. From the latter it extracts a vicinity of the specified
    radius about the coordinates given in the bottom line stats.
    """
    def __init__(self, embedding_dim=128, window=1):
        super().__init__()

        self.embedding = GlyphEmbedding(embedding_dim)
        self.embedding_dim, self.window = embedding_dim, window

    def forward(self, obs):
        k = self.window

        glyphs = obs['glyphs']
        dims = dict(zip('TBRC', glyphs.shape))

        # turn T x B x R x C into T x B x (k+R+k) x (k+C+k) x *F
        #  by pading symmetrically with invalid glyphs and embedding.
        gl_padded = self.embedding(F.pad(glyphs, (k,) * 4, value=MAX_GLYPH))
        # XXX we embed extra 2(R + C) + 4 glyphs, i.e. about 13% overhead

        # unfold into [T x B] x R x C x *F x (k+1+k) x (k+1+k)
        # XXX could we have used einops here?
        n_seq, n_batch, n_rows, n_cols = glyphs.size()  # use unpadded dims
        n_features = gl_padded.shape[4:]
        s_seq, s_batch, s_rows, s_cols, *s_features = gl_padded.stride()

        # use stride trix to dimshiffle and unfold into sliding local windows
        gl_folded = gl_padded.as_strided(
            # flatten T x B, fold R x C into 2d windows in dims after F
            (n_seq * n_batch, n_rows, n_cols, *n_features, k+1+k, k+1+k),
            # when mul-ing dims in shape use the lowest stride
            (s_batch, s_rows, s_cols, *s_features, s_rows, s_cols),
        )

        # extract vicinities around the row-col coordinates, specified in bls
        bls = obs['blstats']
        idx = torch.stack([
            torch.arange(n_seq * n_batch),
            bls[..., NLE_BL_Y].flatten(),
            bls[..., NLE_BL_X].flatten(),
        ], dim=0).T
        gl_vicinity = torch.stack([
            gl_folded[j, r, c] for j, r, c in idx
        ], dim=0).reshape(n_seq, n_batch, *gl_folded.shape[3:])
        # XXX there has to be a better way to pick over dims (0, 1, 2)!

        # now permute the embedded glyphs to T x B x *F x R x C
        gl_permuted = rearrange(gl_padded, 'T B R C ... -> T B ... R C')
        gl_screen = gl_permuted[..., k:-k, k:-k].contiguous()

        # embed inventory glyphs (need to replace NO_GLYPH with MAX_GLYPH,
        #  unless they coincide)
        gl_inventory = rearrange(self.embedding(obs['inv_glyphs']),
                                 'T B N ... -> T B ... N')

        # final touch is to extract the embedding of `self` on the map
        gl_self = gl_vicinity[..., k, k]
        return dict(
            screen=gl_screen,  # already contiguous
            vicinity=gl_vicinity,  # already contiguous
            inventory=gl_inventory.contiguous(),
            self=gl_self.contiguous(),
        )


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
