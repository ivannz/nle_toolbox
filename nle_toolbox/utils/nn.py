import torch
from typing import Optional, Mapping, Any, Union, NamedTuple
from torch.nn import Module, ModuleDict as BaseModuleDict


# copied from rlplay
def onehotbits(
    input: torch.Tensor,
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


class OneHotBits(torch.nn.Module):
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
        input: torch.Tensor,
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
    ):
        # namedtupels are almost like frozen dicts
        if isinstance(input, tuple) and hasattr(type(input), '_fields'):
            input = input._asdict()

        # the same key order as the order of the declaration in  __init__
        apply = {k: m(input[k]) for k, m in self.items()}
        if self.dim is None:
            return apply

        return torch.cat(tuple(apply.values()), dim=self.dim)
