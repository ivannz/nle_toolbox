import plyr

import torch
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from typing import Optional, Any

from .basic import ObsEmbedding, ValPolPair
from ..transformer.hit import HiT
from ...utils.nn import LinearSplitter, ModuleDict
from ...utils.nn import masked_rnn, masked_multinomial


class NLEHITNeuralAgent(nn.Module):
    """A highway transformer agent.
    """
    def __init__(
        self,
        n_context: int,
        embedding_dim: int,
        num_attention_heads: int = 4,
        intermediate_size: int = 16,
        n_cls: int = 4,
        n_io: int = 8,
        head_size: int = None,
        dropout: float = 0.,
        n_layers: int = 1,
        *,
        pol: dict,
        val: dict,
        act: dict = None,
        h0: bool = True,
        tau: dict = None,
    ):
        super().__init__()

        self.features = ModuleDict(dict(
            # XXX ignores kwargs not declared at `__init__` or set to `None`
            obs=ObsEmbedding(embedding_dim),
            act=nn.Sequential(
                nn.Embedding(**act),
                nn.Unflatten(-1, (1, -1,)),  # `.unsqueeze(-2)`
            ) if act else None,
            rew=None,
            fin=None,
        ), dim=-2)

        self.core = HiT(
            n_context + (1 if act else 0),
            embedding_dim,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            n_cls=n_cls,
            n_io=n_io,
            head_size=head_size,
            dropout=dropout,
            n_layers=n_layers,
        )

        if h0:
            self.h0 = nn.Parameter(torch.zeros(self.core.iox.shape))
        else:
            self.register_parameter('h0', None)

        # actor-critic and temperature heads
        self.pol = LinearSplitter(**pol)
        self.val = LinearSplitter(**val)
        self.tau = LinearSplitter(**tau) if tau else None

    @property
    def initial_hx(self) -> Optional[Tensor]:
        return self.h0

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

        if 'action_mask' in obs:
            act = plyr.apply(masked_multinomial, pol, obs['action_mask'])

        else:
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
        n_cls: int = 8,
        n_io: int = 8,
        intermediate_size: int = 64,
        num_attention_heads: int = 4,
        head_size: int = 16,
        *,
        num_layers: int = 1,
        k: int = 3,
        bls: tuple[str] = (
            'hp',
            'hunger',
            'condition',
        ),
        learn_tau: bool = False,
    ) -> dict:
        # compile the recipe (see `HiT`)
        recipe = {
            # 7 x 7 is window of glyphs plus blstats
            'n_context': (k + 1 + k) * (k + 1 + k) + len(bls),
            'embedding_dim': embedding_dim,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'n_cls': n_cls,
            'n_io': n_io,
            'head_size': head_size,
            'dropout': 0.0,
            'n_layers': num_layers,
        }

        # pervious action `act` and reset flag `fin` are optional
        recipe['act'] = {
            'num_embeddings': n_actions,
            'embedding_dim': embedding_dim,
        }

        # join with the kwargs for two `LinearSplitter` heads
        return {
            **recipe,
            'val': {
                'in_features': n_cls * embedding_dim,
                'out_features': {
                    'ext': 1,
                    'int': 1,
                },
            },
            'pol': {
                'in_features': n_cls * embedding_dim,
                'out_features': n_actions,
            },
            'tau': {
                'in_features': n_cls * embedding_dim,
                'out_features': 1,
            } if learn_tau else None,
        }
