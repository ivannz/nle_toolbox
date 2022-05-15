import torch
from torch import nn

from nle.nethack import (
    NLE_BL_HUNGER,
    NLE_BL_CONDITION,
    NLE_BL_HP,
    NLE_BL_HPMAX,
    NLE_BL_ENE,
    NLE_BL_ENEMAX,
    NLE_BL_STR25,
    NLE_BL_STR125,
    NLE_BL_DEX,
    NLE_BL_CON,
    NLE_BL_INT,
    NLE_BL_WIS,
    NLE_BL_CHA,
    NLE_BL_AC,
    NLE_BL_CAP,
)

# botl stats that were not accounted for.
from nle.nethack import (
    # player coords
    NLE_BL_X,
    NLE_BL_Y,
    # in-game score and gold
    NLE_BL_SCORE,
    NLE_BL_GOLD,
    # level and experience
    NLE_BL_XP,
    NLE_BL_EXP,
    # in-game move number
    NLE_BL_TIME,
    # dungeon number, level, and the depth (derived from the fisrt two)
    NLE_BL_DNUM,
    NLE_BL_DLEVEL,
    NLE_BL_DEPTH,
    # 'monster_level' -- the level of the monster when polymorphed
    #  [``](./nle/win/rl/winrl.cc#L552-553)
    NLE_BL_HD,
    # player's alignment
    NLE_BL_ALIGN,
)

from ..utils.nn import OneHotBits, EquispacedEmbedding


class BaseEmbedding(nn.Embedding):
    def __init__(
        self,
        index: int,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.index = index

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        return super().forward(blstats[..., self.index])


class BaseStatEmbedding(BaseEmbedding):
    # for the 6 base stats (luck is hidden) the range is 0..25
    def __init__(self, index: int, embedding_dim: int) -> None:
        super().__init__(index, 25, embedding_dim)


class Hunger(BaseEmbedding):
    from ..utils.env.defs import hunger

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(
            NLE_BL_HUNGER,
            self.hunger.MAX + 1,
            embedding_dim,
            self.hunger.MAX,
        )


class Encumberance(BaseEmbedding):
    from ..utils.env.defs import encumberance

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(
            NLE_BL_CAP,
            self.encumberance.MAX + 1,
            embedding_dim,
            self.encumberance.MAX,
        )


class Condition(nn.Module):
    from ..utils.env.defs import condition

    def __init__(
        self,
        embedding_dim: int,
        *,
        nonlinearity: str = "tanh",
    ) -> None:
        assert nonlinearity in ("tanh", "relu")
        super().__init__()

        # Ordinarily, embeddings do not need non-linearities, due to implicit
        # one-hot encoding. However, here `condition` bottom line stat is
        # a binary flag vector, which defaults to all-zeros, and thus might
        # actually need one, unlike one-hot token embeddings.
        self.onehot = OneHotBits(self.condition.N_BITS)
        self.linear = nn.Linear(
            self.condition.N_BITS,
            embedding_dim,
            bias=True,  # XXX needed for default all-zeros `condition`
        )
        self.nonlinearity = nn.Tanh() if nonlinearity == "tanh" else nn.ReLU()

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        x = self.onehot(blstats[..., NLE_BL_CONDITION])
        return self.nonlinearity(self.linear(x))


class ArmorClass(nn.Embedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(
            24,  # the AC is mapped to 24 bins by the lookup table below
            embedding_dim,
            padding_idx=None,  # no padding index,
        )

        # a bin lookup table for armor_class, a categorical variable.
        self.register_buffer(
            "lut",
            torch.tensor(
                # 0..10 mapped to 11..1, 11..127 to 0
                [*reversed(range(1, 12))]
                + [0] * 117
                # 128..244 mapped to 23, 245..256 to 22..12
                + [23] * 117
                + [*range(22, 11, -1)]
            ),
        )

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        # 'armor_class' in NetHack is descending just like in adnd. In the
        # [code](src/do_wear.c#L2107-2153) it appears that AC is confined
        # to the range of a `signed char`, however to adnd 2e mechanics it
        # is sufficient to consider the range [-10, 10] for the player's AC,
        # since we make d20 rolls anyway.
        # https://merricb.com/2014/06/08/a-look-at-armour-class-in-original-dd-and-first-edition-add/
        # XXX Also, NetHack, just why?! include/hack.h#L499-500
        return super().forward(self.lut[blstats[..., NLE_BL_AC]])


class HP(nn.Module):
    def __init__(self, embedding_dim: int, num_bins: int = 10) -> None:
        super().__init__()
        self.onehot = EquispacedEmbedding(0, 1, steps=num_bins - 1, scale="lin")
        self.linear = nn.Linear(num_bins, embedding_dim, bias=False)

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        hp = blstats[..., NLE_BL_HP] / blstats[..., NLE_BL_HPMAX]
        return self.linear(self.onehot(torch.nan_to_num_(hp)))


class MP(nn.Module):
    def __init__(self, embedding_dim: int, num_bins: int = 10) -> None:
        super().__init__()
        self.onehot = EquispacedEmbedding(0, 1, steps=num_bins - 1, scale="lin")
        self.linear = nn.Linear(num_bins, embedding_dim, bias=False)

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        mp = blstats[..., NLE_BL_ENE] / blstats[..., NLE_BL_ENEMAX]
        return self.linear(self.onehot(torch.nan_to_num_(mp)))


class STR125(nn.Module):
    """blstats data must be preprocessed with `NLEFeatureExtractor` from
    `.utils.env.wrappers`
    """

    def __init__(self, embedding_dim: int, num_bins: int = 10) -> None:
        super().__init__()
        self.onehot = EquispacedEmbedding(0, 1, steps=num_bins - 1, scale="lin")
        self.linear = nn.Linear(num_bins, embedding_dim, bias=False)

    def forward(self, blstats: torch.Tensor) -> torch.Tensor:
        # embedding (adjusted) percentage strength for warrior classes
        prc = blstats[..., NLE_BL_STR125].div(99)
        return self.linear(self.onehot(prc))


class STR(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_STR25, embedding_dim)


class DEX(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_DEX, embedding_dim)


class CON(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_CON, embedding_dim)


class INT(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_INT, embedding_dim)


class WIS(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_WIS, embedding_dim)


class CHA(BaseStatEmbedding):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(NLE_BL_CHA, embedding_dim)
