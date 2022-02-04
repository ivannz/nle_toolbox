"""Probably useful diagnostic tools.
"""
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
    """Compute the number of denormal floats in the parameters of the module.

    Details
    -------
    See [this question](https://stackoverflow.com/questions/36781881) and
    [Performance Issues](https://en.wikipedia.org/wiki/Subnormal_number).
    To summarize, CPU's FPU architectures pick fast pathways for normal
    floats (those which have a non-zero `exp` IEEE754 field), but resort
    to MUCH slower machine code when dealing with denormalized floats.

    If during training on CPU the regularization or soft-sparsification induce
    a large number of denormals, it cloud be remedied by calling

        torch.set_flush_denormal(True)

    before the loop. However, this makes the results of multiplications of
    normal non-zero, but really tiny, floats to be replaced in hardware by
    HARD zeros, which adversely impacts the overall numerical accuracy.
    """

    # bit field sizes of the IEEE754 floating point format
    info = {
        torch.float16: (np.uint16, 10, 5),   # 1 + 5 + 10
        torch.float32: (np.uint32, 23, 8),   # 1 + 8 + 23
        torch.float64: (np.uint64, 52, 11),  # 1 + 11 + 52
    }
    # XXX `.frexp` returns exp and frac, which represent a given float as
    # `(2**exp) * frac`, with frac normalized to (-1, +1). To this end it
    # adjusts the exp value below the normalized fp range (one larger than
    # the ranges given in [the wiki]().)
    # to use it for detecting denormals the returned exp should be compared
    # to the lowest normal exponent.

    for name, par in module.named_parameters():
        dtype, n_frac, n_exp = info[par.dtype]
        # view `float` as an `unsigned int` and fetch the exponent bits
        # XXX makes an expensive copy, unless par is already on HOST (cpu).
        uint = np.asarray(par.detach().cpu()).view(dtype)
        exp = (uint >> n_frac) & ((1 << n_exp) - 1)

        yield name, ((exp == 0).sum(), par.numel())
