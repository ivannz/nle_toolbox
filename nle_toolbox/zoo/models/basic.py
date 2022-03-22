import plyr

import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict, namedtuple

from torch import Tensor
from typing import Optional, Any

from einops import rearrange
from einops.layers.torch import Rearrange

from .. import blstats
from ..glyph import GlyphEmbedding, EgoCentricEmbedding
from ...utils.nn import LinearSplitter, ModuleDict, ParameterContainer
from ...utils.nn import masked_rnn, rnn_hx_shape
from ...utils.nn import apply_mask, masked_multinomial


ValPolPair = namedtuple('ValPolPair', 'val,pol')
ValPolPair.__doc__ = """
Actor-critic's value-policy pair for convenient access and keeping related
data close.
"""


class BoolEmbedding(nn.Embedding):
    """An embedding for the termination/reset flag.
    """
    def __init__(self, embedding_dim: int, **ignore) -> None:
        super().__init__(2, embedding_dim, None, None)

    def forward(self, fin: torch.Tensor) -> torch.Tensor:
        return super().forward(fin.long())


class NonRNNCore(nn.Module):
    """A drop-in non-recurrent replacement for recurrent cores.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        **ignore,
    ) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(input_size, 2 * hidden_size),
            nn.GLU(dim=-1),
        )

    def forward(
        self,
        input: Tensor,
        hx: Any = None,
    ) -> tuple[Tensor, Any]:
        assert hx is None
        return self.body(input), None


class ObsGlyphVicinityEmbedding(EgoCentricEmbedding):
    """The ego-centric vicinity embedding."""
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

        self.glyphs = GlyphEmbedding(embedding_dim)

    def forward(self, vicinity: Tensor, **ignore) -> Tensor:
        # glyph -->> W_e[glyph.entity] + W_g[glyph.group] + ego
        return super().forward(self.glyphs(vicinity))


BLS_EMBEDDERS = {
    'hunger': blstats.Hunger,
    'encumberance': blstats.Encumberance,
    'condition': blstats.Condition,
    'armorclass': blstats.ArmorClass,
    'hp': blstats.HP,
    'mp': blstats.MP,
    'str125': blstats.STR125,
    'str': blstats.STR,
    'dex': blstats.DEX,
    'con': blstats.CON,
    'int': blstats.INT,
    'wis': blstats.WIS,
    'cha': blstats.CHA,
}


class ObsBLStatsEmbedding(nn.ModuleDict):
    """The ModuleDict, that applies stored modules to the input tensor.
    """
    def __init__(
        self,
        embedding_dim: int,
        dim: int,
        stats: tuple[str] = (
            'hp',
            'hunger',
            'condition',
        ),
    ) -> None:
        super().__init__({
            k: BLS_EMBEDDERS[k](embedding_dim) for k in stats
        })

        self.dim = dim

    def forward(self, blstats: Tensor) -> Tensor:
        # the same key order as the order of the declaration in  __init__
        return torch.stack([m(blstats) for m in self.values()], self.dim)


class ObsEmbedding(nn.Module):
    """The combined ego-centric vicinity - inventory embedding.

    The design of a simple obervation encoder:
    * additively embed glyphs' entities and groups
    * employ the `ego` embedding
    * join with embeddings of 'health', 'hunger' and 'condition' form the botl
    """
    def __new__(cls, embedding_dim: int) -> None:
        vicinity = ObsGlyphVicinityEmbedding(embedding_dim)

        return ModuleDict({
            # get the pre-extracted ego-centric vicinities
            'vicinity': nn.Sequential(OrderedDict([
                ('egoglyph', vicinity),
                ('flatten', Rearrange('... H W F -> ... (H W) F')),
            ])),
            # embed the vital stats
            'blstats': ObsBLStatsEmbedding(
                embedding_dim,
                dim=-2,
            ),
            # embed inventory glyphs
            # XXX need to replace NO_GLYPH with MAX_GLYPH, unless they coincide
            # 'inv_glyphs': vicinity.glyphs,
        }, dim=-2)


class SimpleEncoder(nn.Module):
    """Dense layer-based feature encoder for the NLE.
    """
    def __init__(
        self,
        n_context: int,
        embedding_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedding = ObsEmbedding(embedding_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            Rearrange('B N C -> B (N C)'),
            nn.Linear(n_context * embedding_dim,
                      intermediate_size, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, obs: Tensor) -> Tensor:
        x = self.embedding(obs)

        size = dict(zip("TBNF", x.shape[:2]))
        out = self.encoder(rearrange(x, 'T B ... -> (T B) ...'))
        return rearrange(out, '(T B) ... -> T B ...', **size)


class NLENeuralAgent(nn.Module):
    """A generic agent for split encoder-core architecture.

    Details
    -------
    The network feeds the **obs**ervation $x_t$, **act**ion $a_{t-1}$,
    **rew**ard $r_t$, and **fin**ish flag $d_t$ through the provided `features`
    network to get the joint representation of the data from the env. The data
    is then passed into the LSTM core along with the current recurrent state
    `hx` $h_t$ to get the intermediate features and the next state. Finally,
    the intermediate representation, which accumulate historical data from the
    env, are fed into value and policy heads.
    """
    def __init__(
        self,
        obs: dict,
        *,
        core: dict,
        pol: dict,
        val: dict,
        act: dict = None,
        fin: dict = None,
        h0: bool = True,
        tau: dict = None,
    ) -> None:
        super().__init__()

        # our custom ModuleDict ignores kwargs not declared at
        #  `__init__` or those set to `None`.
        self.features = ModuleDict(dict(
            obs=SimpleEncoder(**obs),
            act=nn.Embedding(**act) if act else None,
            rew=None,  # to embed `rew` we'd better disctretize it!
            fin=BoolEmbedding(**fin) if fin else None,
        ), dim=-1)

        # build the core and set up the initial recurrent state
        core = dict(core)  # XXX make a shallow copy, since we'll be popping

        cls = core.pop('cls')
        layer = {'lstm': nn.LSTM, 'gru': nn.GRU, 'linear': NonRNNCore}[cls]
        self.core = layer(**core)

        if h0 and cls != 'linear':
            # Reinterpret the potential nested containers of `h0`
            #  as special `nn.Module`'s parameter containers
            # XXX why should the agent OWN the initial hx? Can we
            #  make it an external parameter?
            h0 = plyr.apply(torch.zeros, rnn_hx_shape(self.core))
            self.h0 = ParameterContainer(plyr.apply(nn.Parameter, h0))

        else:
            self.register_parameter('h0', None)

        # actor-critic and temperature heads
        self.pol = LinearSplitter(**pol)
        self.val = LinearSplitter(**val)
        self.tau = LinearSplitter(**tau) if tau else None

    @property
    def initial_hx(self) -> Optional[Tensor]:
        # See `ParameterContainer` for the `._value` method.
        if self.h0 is not None:
            return self.h0._value()

    def forward(
        self,
        obs: Any,
        act: Any = None,
        rew: Any = None,
        fin: Tensor = None,
        *,
        hx: Any = None,
    ) -> tuple[Any, ValPolPair, Any]:
        # `.features` is a ModuleDict, which ignores kwargs NOT declared
        #  at its `__init__`, which makes `locals()` work really neatly here.
        x = self.features(locals())

        out, hx = masked_rnn(self.core, x, hx, reset=fin, h0=self.initial_hx)

        pol = self.pol(out)

        # apply the temperature
        if self.tau is not None:
            tau = plyr.apply(F.softplus, self.tau(out))
            pol = plyr.apply(torch.mul, pol, tau)

        # It appears that sampling from a masked distribution distabilizes
        #  policy-grad-based algos. This is possibly due to changed support
        #  which this makes everything severely off-policy.
        if 'action_mask' in obs:
            pol = plyr.apply(apply_mask, pol, obs['action_mask'], value=-10.)
            # XXX use a not so extreme filler value

        act = plyr.apply(masked_multinomial, pol, mask=None)

        # act, (val, pol), hx
        return act, ValPolPair(
            plyr.apply(torch.squeeze, self.val(out), dim=-1),
            plyr.apply(F.log_softmax, pol, dim=-1),
        ), hx

    @staticmethod
    def default_recipe(
        n_actions: int,
        embedding_dim: int = 16,
        intermediate_size: int = 256,
        hidden_size: int = 128,
        core: str = 'lstm',
        *,
        act_embedding_dim: int = None,
        fin_embedding_dim: int = None,
        num_layers: int = 2,
        k: int = 3,
        bls: tuple[str] = (
            'hp',
            'hunger',
            'condition',
        ),
        h0: bool = True,
        learn_tau: bool = False,
    ) -> dict:
        # compile the recipe
        recipe = {}

        # `obs` are required (see `SimpleEncoder`)
        recipe['obs'] = {
            # 7 x 7 is window of glyphs plus blstats
            'n_context': (k + 1 + k) * (k + 1 + k) + len(bls),
            'embedding_dim': embedding_dim,
            'intermediate_size': intermediate_size,
            'dropout': 0.0,
        }
        n_core_input_size = intermediate_size

        # pervious action `act` and reset flag `fin` are optional
        if act_embedding_dim is None:
            act_embedding_dim = embedding_dim

        if fin_embedding_dim is None:
            fin_embedding_dim = embedding_dim

        if act_embedding_dim > 0:
            # see `nn.Embedding`
            recipe['act'] = {
                'num_embeddings': n_actions,
                'embedding_dim': act_embedding_dim,
            }
            n_core_input_size += act_embedding_dim

        if fin_embedding_dim > 0:
            # see `BoolEmbedding`
            recipe['fin'] = {
                # `num_embeddings` for bool flags is as always two!
                'embedding_dim': fin_embedding_dim,
            }
            n_core_input_size += fin_embedding_dim

        # add the core to the recipe (See `nn.LSTM` or `NonRNNCore`)
        recipe['core'] = {
            'cls': core,
            'input_size':  n_core_input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
        }

        # join with the kwargs for two `LinearSplitter` heads
        return {
            **recipe,
            'val': {
                'in_features': hidden_size,
                'out_features': {
                    'ext': 1,
                    'int': 1,
                },
            },
            'pol': {
                'in_features': hidden_size,
                'out_features': n_actions,
            },
            'tau': {
                'in_features': hidden_size,
                'out_features': 1,
            } if learn_tau else None,
            'h0': h0,
        }
