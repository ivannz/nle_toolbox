import torch
from torch import nn

from torch.nn import functional as F

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


class GlyphEgoCentricVicinityEmbedding(GlyphEmbedding):
    """Ego-centric embedding for vicinities (centered symmetric glpyh views).
    """
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim)

        # ego embedding offset
        self.ego = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, vicinity):
        # vicinity is `... x (k + 1 + k) x (k + 1 + k)`
        assert vicinity.shape[-1] == vicinity.shape[-2]
        k = vicinity.shape[-1] // 2

        # add ego at the centre of the vicinity
        # XXX  `.pad` pads dims from tail
        glyphs = super().forward(vicinity.long())
        return glyphs.add(F.pad(self.ego, (0, 0,) + (k, k,) * 2))
