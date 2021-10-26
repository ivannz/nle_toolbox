import re
import torch

from ...utils.env.defs import hunger, condition
from ...utils.env.defs import encumberance

from ...utils.env.obs import BLStats
from ...utils.nn import OneHotBits


class BLStatsVitalsEmbedding(torch.nn.Module):
    """Embed Vitals:
    * hitpoints, max_hitpoints
    * energy, max_energy
    * hunger_state, condition
    """
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
    """Embed character build:
    * str, strength_percentage
    * dex, con, int, wis, cha
    * armor_class, carrying_capacity
    """
    def __init__(self, n_features=128):
        super().__init__()

        self.stat = torch.nn.Embedding(
            25 + 1,  # stats range 0..25
            32,
            max_norm=1.,
            norm_type=2,
            scale_grad_by_freq=False,
        )

        self.kind = torch.nn.Embedding(
            6,  # 6 base stats (luck is hidden)
            32,
        )

        # embedding (adjusted) percentage strength for warrior classes
        self.strength_percentage = torch.nn.Linear(
            1,
            32,
            bias=False
        )

        self.encumberance = torch.nn.Embedding(
            encumberance.MAX + 1,
            8,
            padding_idx=encumberance.MAX,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

        # a bin lookup table for armor_class, a categorical variable.
        self.register_buffer(
            'ac_lookup', torch.tensor(
                # 0..10 mapped to 11..1, 11..127 to 0
                [*reversed(range(1, 12))] + [0] * 117
                # 128..244 mapped to 23, 245..256 to 22..12
                + [23] * 117 + [*range(22, 11, -1)]
            )
        )
        self.armor_class = torch.nn.Embedding(
            24,
            8,
            # no padding index,
            max_norm=1.,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.features = torch.nn.Linear(
            32 * 6 + 8 + 8,
            n_features,
            bias=True,
        )

    def forward(self, blstats):
        assert isinstance(blstats, BLStats)
        # deal with
        #  'strength_percentage',
        #  'str', 'dex', 'con', 'int', 'wis', 'cha',

        # this is slooow.
        # XXX maybe we should not share tables between stats?
        #   ... but saving throws?
        stats = torch.stack([
            self.stat(blstats.str), self.stat(blstats.dex),
            self.stat(blstats.con), self.stat(blstats.int),
            self.stat(blstats.wis), self.stat(blstats.cha),
        ], dim=-2) + self.kind.weight

        # adjust strength by the percentage score (integer 0..99)
        strpc = blstats.strength_percentage.div(99).unsqueeze(-1)
        stats[..., 0, :] = stats[..., 0, :] + self.strength_percentage(strpc)

        # 'armor_class' in NetHack is descending just like in adnd. In the
        # [code](src/do_wear.c#L2107-2153) it appears that AC is confined
        # to the range of a `signed char`, however to adnd 2e mechanics it
        # is sufficient to consider the range [-10, 10] for the player's AC,
        # since we make d20 rolls anyway.
        # https://merricb.com/2014/06/08/a-look-at-armour-class-in-original-dd-and-first-edition-add/
        # XXX Also, NetHack, just why?! include/hack.h#L499-500
        armor_class = self.armor_class(self.ac_lookup[blstats.armor_class])

        # 'monster_level' -- the level of the monster when polymorphed
        #  [``](./nle/win/rl/winrl.cc#L552-553)
        # XXX probably should be combined with `self` glyph embedding
        # self.monster_level(blstats.monster_level)

        #  'carrying_capacity'
        carrying_capacity = self.encumberance(blstats.carrying_capacity)
        return self.features(torch.cat([
            stats.flatten(-2, -1),
            carrying_capacity,
            armor_class,
        ], dim=-1))


class BLStatsEmbedding(torch.nn.Module):
    """Embed Bottom Line Stats:
    * str, strength_percentage, dex, con, int, wis, cha
    * hitpoints, max_hitpoints
    * energy, max_energy
    * armor_class
    * hunger_state
    * carrying_capacity
    * condition
    """
    def __init__(self, n_vitals=128, n_build=32):
        super().__init__()
        self.vitals = BLStatsVitalsEmbedding(n_vitals)
        self.build = BLStatsBuildEmbedding(n_build)

    def forward(self, obs):
        # turn blstats into a namedtuple
        bls = BLStats(*obs['blstats'].unbind(-1))

        return torch.cat([
            self.vitals(bls),
            self.build(bls),
        ], dim=-1)

        # what do we do with these?
        # x, y, score
        # 'gold'  # XXX useful for shops
        # 'depth', 'dungeon_number', 'level_number'
        # XXX 'depth' is determined by 'level_number' and 'dungeon_number'
        # monster_level, experience_level, experience_points
        # 'time'  # XXX is the game time counter useful at all?


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
