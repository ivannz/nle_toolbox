import torch
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
        grp, ent = torch.unbind(self.gl_lookup[glyphs], dim=-1)
        return torch.cat([self.entity(ent), self.group(grp)], dim=-1)


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
