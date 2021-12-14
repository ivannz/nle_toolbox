import torch
from torch import nn

from torch import Tensor
from typing import Optional, Any, List, Tuple, Mapping, Union

from copy import deepcopy
from rlplay.engine.utils import plyr

from .glyph import GlyphFeatures
from .blstats import BLStatsEmbedding

from ...utils.nn import ModuleDict, LinearSplitter
from ...utils.nn import hx_shape, hx_broadcast


class NetworkFeatures(nn.Module):
    """Neural block that extracts features form the NLE's observations."""
    def __init__(
        self,
        sizes: List[int],
        glyphs: Mapping[str, Any],
        bls: Mapping[str, Any],
    ):
        super().__init__()

        # the embedders
        self.glyphs = GlyphFeatures(**glyphs)
        self.bls = BLStatsEmbedding(**bls)

        # build the feature network
        layers = [
            ModuleDict(dict(
                bls=nn.Identity(),
                vicinity=nn.Flatten(-3, -1),
                inventory=nn.Flatten(-2, -1),
            ), dim=-1)
        ]

        for n, m in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(n, m, bias=True))
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

    def forward(
        self,
        obs: Mapping[str, Tensor],
    ):
        return self.features(dict(
            bls=self.bls(obs),
            **self.glyphs(obs),
        ))

    @staticmethod
    def validate(
        recipe: Mapping[str, Any],
        sizes: List[int],
        bls: Mapping[str, Any],
        glyphs: Mapping[str, Any],
    ):
        assert all(k in recipe for k in (
            'glyphs', 'bls', 'sizes',
        ))

        return recipe['sizes'][-1]


class NetworkCore(nn.Module):
    """A Recurrent Stack with learnable initial hidden state vector.

    Parameters
    ----------
    input_size: int
        The number of expected features in the input `x`

    hidden_size: int
        The number of features in the hidden state `h`

    num_layers: int, default=1
        Number of layers in the recurrent stack.

    bias: bool, default=True
        If ``False``, then the layers do not use bias weights.

    dropout: float, default=0.
        If non-zero, introduces a `Dropout` layer on the outputs of each
        recurrent layer in the stack except the last layer, with dropout
        probability equal to `dropout`.

    Details
    -------
    This is actually as wrapper around a homogeneous stack of recurrent cores.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.,
        *,
        kind: str = 'gru',
        learnable: bool = True,
    ):
        assert kind in ('lstm', 'gru', 'identity')

        layer = nn.Identity
        if kind == 'gru':
            layer = nn.GRU

        elif kind == 'lstm':
            layer = nn.LSTM

        super().__init__()

        self.core = layer(
            input_size,   # input
            hidden_size,  # hidden
            num_layers,   # height of the stack of cores
            bias=bias,
            dropout=dropout,
            batch_first=False,
            bidirectional=False,
        )

        # construct `h0` for the core
        # XXX implement and use Buffer(List|Dict) for non-learnable initial
        #  state vector.
        h0 = None
        shape = hx_shape(self.core)
        if isinstance(shape, torch.Size):
            h0 = torch.nn.Parameter(torch.zeros(*shape))

        elif isinstance(shape, tuple):
            h0 = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(*s)) for s in shape
            ])

        if h0 is None:
            self.register_parameter('h0', h0)

        else:
            # XXX this may get overridden by the module's `.requires_grad_`
            self.h0 = h0.requires_grad_(learnable)

    @torch.no_grad()
    def reset(
        self,
        hx: Union[Tensor, Tuple[Tensor]],
        at: int,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """Non-differentiably reset the state at the specified batch element.
        """
        # no recurent state to copy from, keep `hx` intact
        if isinstance(self.core, nn.Identity):
            return hx

        if isinstance(at, int):
            at = slice(at, at+1) if at != -1 else slice(at, None)

        elif not isinstance(at, slice):
            raise TypeError(f'Unsupported index `{at}`.')

        # `h0` is `n_states x 1 x *hidden` and `hx` has `n_batch` instead of 1`
        h0 = self.h0
        if isinstance(h0, nn.ParameterList):
            h0 = tuple(h0)  # XXX this iterates over the ParList

        elif not isinstance(h0, nn.Parameter):
            raise TypeError(f'Unsupported initial state type `{self.h0}`.')
        # XXX is there a way to avoid this clunkiness and repetition of 
        #  `hx_broadcast`-s logic here?

        # make a deep copy, then non-differentiably overwrite
        #  with the inital recurrent state data
        hx_ = plyr.suply(torch.Tensor.clone, hx)
        plyr.suply(lambda x, u: x[:, at].copy_(u), hx_, h0)
        # XXX lambda-ing is slow af, but we need to access the second dim.

        return hx_

    def forward(
        self,
        input: Tensor,
        fin: Optional[Tensor] = None,
        hx: Optional[Union[Tensor, Tuple[Tensor]]] = None,
    ) -> Tuple[Mapping[str, Tensor], Union[Tensor, Tuple[Tensor]]]:
        # XXX this entire `.forward` could be factored into a dedicated
        #  masked seq rnn runner (see `psychic-octo-spoon`)

        # run the core
        if isinstance(self.core, nn.Identity):
            return input, ()

        # prepare the hiddens: we keep an explicit `h0`, in the case
        # it is diff-able and learnable
        n_seq, n_batch = input.shape[:2]
        h0 = hx_broadcast(self.h0, n_batch)

        # in case `fin` is missing or all-false, use a faster branch
        hx = h0 if hx is None else hx  # init hx by aliasing h0
        if fin is None or not fin.any():
            return self.core(input, hx=hx)

        # check the leading dims (seq x batch)
        if fin.shape != input.shape[:2]:
            raise ValueError(f'Dim mismatch {fin.shape} != {input.shape[:2]}')

        # make sure the termination mask is numeric for lerping
        fin = fin.unsqueeze(-1).to(input)

        # loop along the `n_seq` dim, but in slices with fake T=1
        outputs = []
        for x, f in zip(input.unsqueeze(1), fin.unsqueeze(1)):
            # reset the hiddens by lerping with `fin`
            if hx is not h0:  # skip if hx has just been initted
                # hx <<-- hx * (1 - f) + h0 * f "==" h0 if f else hx
                hx = plyr.suply(torch.lerp, hx, h0, weight=f)

            out, hx = self.core(x, hx=hx)
            outputs.append(out)

        return torch.cat(outputs, dim=0), hx


class NetworkHead(nn.Module):
    """The multi-head action module with extra value and hlating outputs."""
    def __new__(
        cls,
        n_features: int,
        heads: Mapping[str, int],
    ):
        return LinearSplitter(
            n_features,
            out_features=dict(val=1, hlt=1, **heads),
            bias=True,
        )


class Network(nn.Module):
    """The NLE actor network."""
    def __init__(
        self,
        recipe: Mapping[str, Any],
    ):
        recipe = deepcopy(recipe)
        super().__init__()

        # the feature extractor
        self.features = NetworkFeatures(**recipe['features'])

        # the core, either recurrent of bypass
        self.core = NetworkCore(**recipe['core'])

        # the critic's value, halting logit, raw action scores, etc
        self.head = NetworkHead(**recipe['head'])

    def reset(self, hx, at):
        return self.core.reset(hx, at=at)

    def forward(
        self,
        obs: Mapping[str, Tensor],
        fin: Optional[Tensor] = None,
        hx: Optional[Union[Tensor, Tuple[Tensor]]] = None,
    ) -> Tuple[Mapping[str, Tensor], Union[Tensor, Tuple[Tensor]]]:
        x = self.features(obs)
        out, hx = self.core(x, fin=fin, hx=hx)
        return self.head(out), hx
