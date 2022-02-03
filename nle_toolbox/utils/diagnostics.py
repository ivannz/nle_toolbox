import numpy as np
from time import monotonic_ns

import torch
from torch import nn


class DiagnosticReLU(nn.ReLU):
    """A ReLU activation layer which tracks the output sparisty rate.
    """

    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.alive = []

    def forward(self, input):
        output = super().forward(input)

        # count the share of `alive` outputs
        self.alive.append(int(output.gt(0.).sum()) / output.numel())

        return output


def collect_relu_death(module):
    relu_death = {}
    for nom, mod in module.named_modules():
        if isinstance(mod, DiagnosticReLU):
            assert nom not in relu_death
            relu_death[nom] = 1. - np.array(mod.alive)

    return relu_death


class DiagnosticSequential(nn.Sequential):
    """A Sequential container which track forward pass timestamps.
    """

    def __init__(self, *args: nn.Module):
        super().__init__(*args)
        self.timings_ns = []

    def forward(self, input):
        timing_ns = []
        timing_ns.append(monotonic_ns())  # base

        for module in self:
            input = module(input)
            timing_ns.append(monotonic_ns())  # step

        self.timings_ns.append(tuple(timing_ns))
        return input


def collect_seq_timings(module):
    seq_timings = {}
    for nom, mod in module.named_modules():
        if isinstance(mod, DiagnosticSequential):
            assert nom not in seq_timings

            seq_timings[nom] = np.array(mod.timings_ns), [
                f'({j}) {mod.__class__.__name__}' for j, mod in enumerate(mod)
            ]

    return seq_timings


def named_denormal_stats(module):
    """Compute the number of denormals in the parametrs of the given module.
    """

    # XXX `.frexp` returns exp and fraction to represent the given float.
    # to use it for detecting denormals the returned exp should be compared
    # to the lowest normal exponent.
    shifts = {
        torch.float16: (np.uint16, 10, (1<<5) - 1),
        torch.float32: (np.uint32, 23, (1<<8) - 1),
        torch.float64: (np.uint64, 52, (1<<11) - 1),
    }
    for name, par in module.named_parameters():
        dtype, shift, mask = shifts[par.dtype]
        # view as fp32 as `uint32` and extract the IEEE754 exponent bits
        uint = np.asarray(par.detach()).view(dtype)
        exp = (uint >> shift) & mask
        yield name, ((exp == 0).sum(), par.numel())
