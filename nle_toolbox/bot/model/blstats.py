import re
import torch

from ...utils.env.defs import hunger, condition
from ...utils.env.defs import encumberance

from ...utils.env.obs import BLStats
from ...utils.nn import OneHotBits


class BLStatsVitalsEmbedding(torch.nn.Module):
    """Glyph Embedding is the shared representation layer."""
    def __init__(self, n_features=128):
        super().__init__()

        # hunger state Embedding
        self.hunger = torch.nn.Embedding(
            hunger.MAX + 1,
            16,
            padding_idx=hunger.MAX,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

        # multiple conditions may affect the agent (up to 2^13)
        # XXX we may potentially want to split into three risk levels (defs.py)
        self.status = OneHotBits(condition.N_BITS)

        # hunger | status | health | energy
        self.features = torch.nn.Linear(
            16 + condition.N_BITS + 1 + 1,
            n_features,
            bias=True,
        )

    def forward(self, blstats):
        assert isinstance(blstats, BLStats)
        # vitals
        #  'hunger_state',
        #  'condition',
        #  'hitpoints', 'max_hitpoints',
        hp = torch.nan_to_num_(blstats.hitpoints / blstats.max_hitpoints)
        mp = torch.nan_to_num_(blstats.energy / blstats.max_energy)
        #  'energy', 'max_energy',
        return self.features(torch.cat([
            self.hunger(blstats.hunger_state),
            self.status(blstats.condition),
            torch.unsqueeze(hp - 0.5, dim=-1,),
            torch.unsqueeze(mp - 0.5, dim=-1,),
        ], dim=-1))


class BLStatsBuildEmbedding(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()

        self.stat = torch.nn.Embedding(
            25,  # stats range 0..25
            32,
            max_norm=1.,
            norm_type=2,
            scale_grad_by_freq=False,
        )

        self.kind = torch.nn.Embedding(
            6,  # 6 basic stats (luck is hidden)
            32,
        )

        # embedding (adjusted) percentage strength for warrior classes
        self.strength_percentage = torch.nn.Linear(
            1,
            32,
            bias=False
        )

        self.features = torch.nn.Linear(32 * 6, n_features, bias=True)

    def forward(self, blstats):
        assert isinstance(blstats, BLStats)
        # deal with
        #  'strength_percentage',
        #  'str', 'dex', 'con', 'int', 'wis', 'cha',

        # this is slooow.
        # XXX maybe we should not share tables between stats?
        #   ... but saving throws?
        out = torch.stack([
            self.stat(blstats.str), self.stat(blstats.dex),
            self.stat(blstats.con), self.stat(blstats.int),
            self.stat(blstats.wis), self.stat(blstats.cha),
        ], dim=-2) + self.kind.weight

        # adjust strength by the percentage score (integer 0..99)
        strpc = blstats.strength_percentage.div(99).unsqueeze(-1)
        out[..., 0, :] = out[..., 0, :] + self.strength_percentage(strpc)

        return self.features(out.flatten(-2, -1))


class BLStatsEmbedding(torch.nn.Module):
    """Glyph Embedding is the shared representation layer."""
    def __init__(self):
        super().__init__()
        self.vitals = BLStatsVitalsEmbedding()
        self.build = BLStatsBuildEmbedding()

        self.encumberance = torch.nn.Embedding(
            encumberance.MAX,
            8,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

        # self.armor_class = torch.nn.Embedding(...)

    def forward(self, blstats):
        assert isinstance(blstats, BLStats)
        # 'armor_class' in NetHack is descending just like in adnd. In the
        # [code](src/do_wear.c#L2107-2153) it appears that AC is confined
        # to the range of a `signed char`, however to adnd 2e mechanics it
        # is sufficient to consider the range [-10, 10] for the player's AC,
        # since we make d20 rolls anyway.
        # https://merricb.com/2014/06/08/a-look-at-armour-class-in-original-dd-and-first-edition-add/
        # XXX Also, NetHack, just why?! include/hack.h#L499-500
        #  'armor_class',
        pass

        #  'carrying_capacity',
        return torch.cat([
            self.vitals(blstats),
            self.build(blstats),
            self.encumberance(blstats.carrying_capacity),
        ], dim=-1)

        raise NotImplementedError

        # what do we do with these? 
        #  'x', 'y',
        #  'score', 'gold',
        #  'experience_level', 'experience_points',
        #  'depth', 'level_number', 'monster_level', 'dungeon_number',
        #  'time',  # step counter


# src/allmain.c#L681-682 -- always assume new game
rx_greeting = re.compile(
    r"""
        \s+NetHack!\s+You\s+are\s+a
        # src/priest.c#L874-889 -- alignment strings
        \s+(?P<align>
            chaotic
            |neutral
            |lawful
            |unaligned
            |unknown
        )
        # src/role.c#L753-758 -- genders
        (\s+(?P<gender>
            male
            |female
            |neuter
        ))?
        # src/role.c#L679-726 -- races
        (\s+(?P<race>
            human
            |elven
            |dwarven
            |gnomish
            |orcish
        ))
        # src/role.c#L8-586 -- clases
        (\s+(?P<role>
            archeologist
            |barbarian
            |cave(?P<special1>wo)?man
            |healer
            |knight
            |monk
            |priest(?P<special2>ess)?
            |rogue
            |ranger
            |samurai
            |tourist
            |valkyrie
            |wizard
        ))
    """,
    re.IGNORECASE | re.VERBOSE | re.ASCII,
)


def parse_welcome(message):
    match = rx_greeting.search(message.view('S256')[0].decode('ascii'))
    role = {k: v.lower() for k, v in match.groupdict('').items()}

    # replace the adjective form with the nominal form
    race = {
        "human": "human",
        "elven": "elf",
        "dwarven": "dwarf",
        "gnomish": "gnome",
        "orcish": "orc",
    }[role.pop('race')]

    # gender is not printed in classes which have special female names
    special = role.pop('special1') or role.pop('special2')
    gender = role.pop('gender') or ('female' if special else 'male')
    return {**role, 'gender': gender, 'race': race}
