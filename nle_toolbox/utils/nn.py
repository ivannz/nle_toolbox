import torch
import plyr

from torch import Tensor
from typing import Optional, Mapping, Any, Union, NamedTuple

from torch.nn import Module, ModuleDict as BaseModuleDict
from torch.nn import Linear, LSTM, GRU, RNNBase
from torch.nn import ParameterList, Parameter


# copied from rlplay
def onehotbits(
    input: Tensor,
    n_bits: int = 63,
    dtype: torch.dtype = torch.float,
):
    """Encode integers to fixed-width binary floating point vectors"""
    assert not input.dtype.is_floating_point
    assert 0 < n_bits < 64  # torch.int64 is signed, so 64-1 bits max

    # n_bits = {torch.int64: 63, torch.int32: 31, torch.int16: 15, torch.int8 : 7}

    # get mask of set bits
    pow2 = torch.tensor([1 << j for j in range(n_bits)]).to(input.device)
    x = input.unsqueeze(-1).bitwise_and(pow2).to(bool)

    # upcast bool to float to get one-hot
    return x.to(dtype)


class OneHotBits(Module):
    """Bitfield one-hot encoder."""
    def __init__(
        self,
        n_bits: int = 63,
        dtype: torch.dtype = torch.float,
    ):
        assert 1 <= n_bits < 64
        super().__init__()
        self.n_bits, self.dtype = n_bits, dtype

    def forward(
        self,
        input: Tensor,
    ):
        return onehotbits(input, n_bits=self.n_bits, dtype=self.dtype)


class ModuleDict(BaseModuleDict):
    """The ModuleDict, that applies itself to the input dicts."""
    def __init__(
        self,
        modules: Optional[Mapping[str, Module]] = None,
        dim: Optional[int] = None,
    ) -> None:
        super().__init__(modules)
        self.dim = dim

    def forward(
        self,
        input: Union[Mapping[str, Any], NamedTuple],
    ) -> Union[Mapping[str, Tensor], Tensor]:
        # namedtupels are almost like frozen dicts
        if isinstance(input, tuple) and hasattr(type(input), '_fields'):
            input = input._asdict()

        # the same key order as the order of the declaration in  __init__
        apply = {k: m(input[k]) for k, m in self.items()}
        if self.dim is None:
            return apply

        return torch.cat(tuple(apply.values()), dim=self.dim)


def select(tensor, index, dim, *, at=None):
    """Unbatched select.
    """
    at = dim if at is None else at

    # figure out the axes
    dim = tensor.ndim + dim if dim < 0 else dim
    at = tensor.ndim + at if at < 0 else at

    # dimshuffle if at is not dim
    if at != dim:
        dims = *range(dim), *range(dim+1, tensor.ndim)
        tensor = tensor.permute(*dims[:at], dim, *dims[at:])
        dim = at

    # select and reshape
    shape = tensor.shape[:dim] + index.shape + tensor.shape[dim+1:]
    return tensor.index_select(dim, index.flatten()).reshape(shape)


def bselect(tensor, *index, dim):
    r"""Batched index select over a span of dims starting from `dim`.

    Details
    -------
    Let $x$ be a tensor $
        \mathcal{X}^{d_0 \times \cdots \times d_{m-1}}
    $ and $
        k \in \{0, \cdots, m-1\}
    $ -- some dimension. Given an indexing tensor $
        \iota \in \bigl(
            \prod_{j=0}^{p-1} [d_{k+j}]
        \bigr)^{d_0 \times \cdots \times d_{k-1}}
    $ for some $p\geq 1$ this procedure returns a tensor $
        y \in \mathcal{X}^{
            d_0 \times \cdots \times d_{k-1}
            \times
            d_{k+p} \times \cdots \times d_{m-1}
        }
    $ constructed from batched selection from $x$ using $\iota$:
    $$
        y_\alpha
            = x_{\alpha_{:k} \iota_{\alpha_{:k}} \alpha_{k:}}
            \,, $$
    where $
        \alpha_{:k}
            = \alpha_0 \cdots \alpha_{k-1}
    $, $
        \alpha
            = \alpha_{:k} \alpha_{k:}
            \in \prod_{j=0}^{k-1} [d_j]
            \times \prod_{j=k+p}^{m-1} [d_j]
    $, and $
        \iota_{\alpha_{:k}}
            \in \prod_{j=0}^{p-1} [d_{k+j}]
    $.
    """
    dim = tensor.ndim + dim if dim < 0 else dim

    assert index and all(index[0].shape == idx.shape for idx in index)
    assert tensor.shape[:dim] == index[0].shape[:dim]
    assert tensor.ndim >= dim + len(index)

    lead = tensor.shape[:dim]
    dims = tensor.shape[dim:dim + len(index)]
    tail = tensor.shape[dim + len(index):]

    # ravel mutliindex using an empty tensor
    mock = tensor.new_empty(dims + (0,))
    idx = sum(j * s for j, s in zip(index, mock.stride()))

    # flatten the `tail` and `dims`, keeping `lead` intact
    flat = tensor.flatten(dim + len(index), -1).flatten(dim, -2)

    # gather the slices from the tensor
    idx = idx.view(lead + (1, 1,))
    out = torch.gather(flat, -2, idx.expand(*lead, 1, flat.shape[-1]))

    # tensor[..., *index, ...]
    return out.reshape(*lead, *tail)


class LinearSplitter(Linear):
    """A linear layer splitting its result into a dict of tensors."""

    def __new__(
        cls,
        in_features: int,
        out_features: Union[int, Mapping[str, int]],
        bias: bool = True,
    ):
        # regress back to Linear if `out_features` is unstructured
        if isinstance(out_features, int):
            return Linear(in_features, out_features, bias=bias)

        return object.__new__(cls)

    def __init__(
        self,
        in_features: int,
        out_features: Union[int, Mapping[str, int]],
        bias: bool = True,
    ) -> None:
        n_out = sum(out_features.values())
        super().__init__(in_features, n_out, bias=bias)

        self.names, self.sizes = zip(*out_features.items())

    def forward(
        self,
        input: Tensor,
    ) -> Mapping[str, Tensor]:
        # apply the linear transformation
        out = super().forward(input)

        # then split the outputs along the last dim and give them names
        return dict(zip(self.names, out.split(self.sizes, dim=-1)))

    def extra_repr(self) -> str:
        splits = dict(zip(self.names, self.sizes))

        text = "in_features={}, out_features={}, bias={}"
        return text.format(self.in_features, splits, self.bias is not None)


def hx_shape(self):
    """Get the batch-broadcastible shape of the recurrent state.

    Complaint
    ---------
    Torch devs should've made this into a recurrent nn.Module's method.
    """
    if not isinstance(self, (RNNBase, GRU, LSTM)):
        return None

    n_hidden_states = self.num_layers * (2 if self.bidirectional else 1)

    # order matters due to possible subclassing!
    if isinstance(self, GRU):
        # copied almost verbatim from `torch.nn.GRU.forward`
        return torch.Size((n_hidden_states, 1, self.hidden_size))

    elif isinstance(self, LSTM):
        # copied almost verbatim from `torch.nn.LSTM.forward`
        return (
            torch.Size((
                n_hidden_states,
                1,
                self.proj_size if self.proj_size > 0 else self.hidden_size,
            )),
            torch.Size((n_hidden_states, 1, self.hidden_size)),
        )

    elif isinstance(self, RNNBase):
        # copied almost verbatim from `torch.nn.RNNBase.forward`
        return torch.Size((n_hidden_states, 1, self.hidden_size))

    # raise TypeError


def hx_broadcast(h0, n_batch):
    """Broadcast the initial state h0 to a batch of size `n_batch`."""
    # XXX this procedure could be replaced by
    #   `plyr.suply(torch.tile, h0, dims=(1, n_batch, 1))`
    # but then we need to preprocess the inputs into `dict` or `list`.

    if isinstance(h0, ParameterList):
        return tuple([h.repeat(1, n_batch, 1) for h in h0])

    elif isinstance(h0, Parameter):
        return h0.repeat(1, n_batch, 1)

    return ()
