"""A potpurri of helpful procedures, layers, embedders, and initializers.

This module implements various helper procedures and layers for easier
prototyping of deep networks operating on many-tensor inputs, and training
of and inference with recurrent networks over sequential data with auxiliary
masks, indicating if a pre-forward reset of the hidden recurrent runtime
states to their initial values is in order.

In addition, there are some routines here, that generalize their generic
analogues from torch: a more multidim-friendly versions of `torch.multinomial`
and `torch.index_select` (see `nn.select`), and a batched version of
`torch.gather` (see `nn.bselect`).
"""
import math
import weakref

import torch
import plyr

from torch import Tensor
from typing import Optional, Mapping, Any, Union, NamedTuple

from torch.nn import init
from torch.nn import Module, ModuleDict as BaseModuleDict
from torch.nn import LSTM, GRU, RNN, RNNBase
from torch.nn import Linear, Identity
from torch.nn import Parameter, ParameterList


# copied from rlplay
def onehotbits(
    input: Tensor,
    n_bits: int = 63,
    dtype: torch.dtype = torch.float,
) -> Tensor:
    """Encode integers to fixed-width binary floating point vectors."""
    assert not input.dtype.is_floating_point
    assert 0 < n_bits < 64  # torch.int64 is signed, so 64-1 bits max

    # n_bits = {torch.int64: 63, torch.int32: 31,
    #           torch.int16: 15, torch.int8 : 7}

    # get mask of set bits
    pow2 = torch.tensor([1 << j for j in range(n_bits)]).to(input.device)
    x = input.unsqueeze(-1).bitwise_and(pow2).to(bool)

    # upcast bool to float to get one-hot
    return x.to(dtype)


class OneHotBits(Module):
    """One-hot encoder of bit fields."""

    def __init__(
        self,
        n_bits: int = 63,
        dtype: torch.dtype = torch.float,
    ) -> None:
        assert 1 <= n_bits < 64
        super().__init__()
        self.n_bits, self.dtype = n_bits, dtype

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return onehotbits(input, n_bits=self.n_bits, dtype=self.dtype)

    def extra_repr(self) -> str:
        return "n_bits={n_bits}".format(**vars(self))


class EquispacedEmbedding(Module):
    """One-hot encode the index of a real-valued feature within the specified
    equispaced bins.
    """

    def __init__(
        self,
        start: float,
        end: float,
        steps: int,
        *,
        scale: str = "lin",
        base: float = 10.0,
    ) -> None:
        if scale == "lin":
            breaks = torch.linspace(start, end, steps)

        elif scale == "log":
            breaks = torch.logspace(start, end, steps, base=base)

        else:
            raise ValueError(f"`scale` not in ['lin', 'log']. Got '{scale}'.")

        super().__init__()

        limits = breaks.new_tensor([float("-inf"), float("+inf")])
        self.register_buffer(
            "breaks",
            torch.cat(
                [
                    limits[:1],
                    breaks,
                    limits[1:],
                ]
            ),
        )

        self.start = start
        self.end = end
        self.steps = steps
        self.scale = scale
        self.base = base

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        x = input.unsqueeze(-1)
        return torch.logical_and(self.breaks[:-1] <= x, x < self.breaks[1:],).to(
            input
        )  # XXX to match input's dtype

    def extra_repr(self) -> str:
        fmt = "{start}, {end}, {steps}, scale={scale}"
        if self.scale == "log":
            fmt += ", base={base}"

        return fmt.format(**vars(self))


class ModuleDict(BaseModuleDict):
    """The ModuleDict, that applies itself to the data in the input dict.

    Details
    -------
    The keys/fields, that are NOT DECLARED at `__init__`, are silently
    IGNORED and filtered out by `.forward`.
    """

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
        # namedtuples are almost like frozen dicts
        if isinstance(input, tuple) and hasattr(type(input), "_fields"):
            input = input._asdict()

        # the same key order as the order of the declaration in  __init__
        apply = {k: m(input[k]) for k, m in self.items() if m is not None}
        if self.dim is None:
            return apply

        return torch.cat(tuple(apply.values()), dim=self.dim)


def select(tensor, index, dim, *, at=None):
    """Unbatched select.

    Returns a new tensor which indexes the `input` tensor along dimension
    `dim` using the entries in `index` which is a `LongTensor`.
    """
    at = dim if at is None else at

    # figure out the axes
    dim = tensor.ndim + dim if dim < 0 else dim
    at = tensor.ndim + at if at < 0 else at

    # dimshuffle if at is not dim
    if at != dim:
        dims = *range(dim), *range(dim + 1, tensor.ndim)
        tensor = tensor.permute(*dims[:at], dim, *dims[at:])
        dim = at

    # select and reshape
    shape = tensor.shape[:dim] + index.shape + tensor.shape[dim + 1 :]
    return tensor.index_select(dim, index.flatten()).reshape(shape)


def bselect(tensor, *index, dim):
    r"""Batched index select over a span of dims starting from `dim`.

    Unlike `torch.gather` for an at least 4-D input with dim=2 and
    index = (ix_0, ix_1, ix_2) tensor the output is specified by:

        out[i, j, ...] = input[i, j, ix_0[i, j], ix_1[i, j], ix_2[i, j], ...]

    where `i, j` (dim=2) are treated as batch dimension indices, and each
    `ix_k` addresses the corresponding dimension `dim+k` in `input`.

    `troch.garther` requires that the `index` have as many dims as the `input`,
    so it cannot select, for example, patches in from a batch of images
    at a batch of coordinates.

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
    $ for some $p \geq 1$, this procedure returns a tensor $
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
    dims = tensor.shape[dim : dim + len(index)]
    tail = tensor.shape[dim + len(index) :]

    # ravel multiindex using an empty tensor
    mock = tensor.new_empty(dims + (0,))
    idx = sum(j * s for j, s in zip(index, mock.stride()))

    # flatten the `tail` and `dims`, keeping `lead` intact
    flat = tensor.flatten(dim + len(index), -1).flatten(dim, -2)

    # gather the slices from the tensor
    idx = idx.view(
        lead
        + (
            1,
            1,
        )
    )
    out = torch.gather(flat, -2, idx.expand(*lead, 1, flat.shape[-1]))

    # tensor[..., *index, ...]
    return out.reshape(*lead, *tail)


class LinearSplitter(Linear):
    """A linear layer splitting its result into a dict of tensors.

    Details
    -------
    Operates much like `nn.Linear`, except this one accepts a non-nested
    dict of ints in `out_features`, that specifies the number of outputs,
    dedicated to each key. REGRESSES to `nn.Linear` in the case when
    `out_features` is just an int.
    """

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
        # `out_features` is a potentially nested object, so we extract
        #  its structre and flatten the size data contained within
        sizes, structure = plyr.flatten(out_features)
        if not all(isinstance(sz, int) and sz >= 0 for sz in sizes):
            raise ValueError("Structured output sizes cannot be negative.")

        super().__init__(in_features, sum(sizes), bias=bias)
        self.structure, self.sizes = structure, tuple(sizes)

    def forward(
        self,
        input: Tensor,
    ) -> Mapping[str, Tensor]:
        # apply the linear layer and split the outputs along the last dim
        flat = torch.split(super().forward(input), self.sizes, dim=-1)

        # ... then rebuild the correct nested structure
        return plyr.unflatten(flat, self.structure)

    def extra_repr(self) -> str:
        splits = plyr.unflatten(self.sizes, self.structure)

        text = "in_features={}, out_features={}, bias={}"
        return text.format(self.in_features, splits, self.bias is not None)


class Splitter(Identity):
    """A layer that splits the input into a dict of tensors."""

    def __init__(
        self,
        splits: Union[int, Mapping[str, int]],
        dim: int = -1,
    ) -> None:
        # `splits` is a potentially nested object, so we extract
        #  its structre and flatten the size data contained within
        sizes, structure = plyr.flatten(splits)
        if not all(isinstance(sz, int) and sz >= 0 for sz in sizes):
            raise ValueError("Structured output sizes cannot be negative.")

        super().__init__()
        self.splits, self.dim = splits, dim
        self.structure, self.sizes = structure, tuple(sizes)

    def forward(self, input: Tensor) -> Mapping[str, Tensor]:
        # apply the linear layer and split the outputs along the last dim
        flat = input.split(self.sizes, dim=self.dim)

        # ... then rebuild the correct nested structure
        return plyr.unflatten(flat, self.structure)

    def extra_repr(self) -> str:
        return repr(plyr.unflatten(self.sizes, self.structure))


class ModuleDictSplitter(BaseModuleDict):
    """A dictionary that applies the contained modules to the input tensor."""

    def forward(self, input: Tensor) -> Mapping[str, Tensor]:
        return {k: None if m is None else m(input) for k, m in self.items()}


def scale_grad(x: Tensor, scale: Union[float, Tensor] = 0.5) -> Tensor:
    """Scale the grads for the backprop."""

    # We can scale grads from less reliable estimates, or longer trajectories
    # to prevent exploding recurrent grads (making h-h vanish instead)...
    # XXX Mu-zero does this in its `pseudocode.py::update_weights` and in
    #  the Nature paper caliming that
    #  """[To] maintain roughly similar magnitude of gradient across different
    #  unroll steps [... we] scale the loss of each head by [...] the number
    #  of [steps to ensure] that the total gradient has similar magnitude
    #  irrespective of how many steps we unroll for[, and also] scale the
    #  gradient at [every application] of the dynamics function by 1/2
    #  [ensuring] that the total gradient  applied to [it] stays constant."""
    #  (Schrittweiser et al.; 2020 Nature, p. 9, left column top)

    # compute `x -->> (1 - s) * [x] + s * x` which identically evals to `x`,
    #  but has diffablility properties of `s * x`!
    return x.detach().lerp(x, scale)  # XXX a.lerp(b, w) = a + w * (b - a)


class InputGradScaler(Identity):
    r"""Scale the gradients passing through this layer without affecting
    the values.

    Details
    -------
    Let $\operatorname{sg}$ be the \emph{stop-grad} operator, which makes
    its argument a constant. In order to neatly implement input grad scaling
    we use the following stop-grad trick. For $\eta \in \mathbb{R}$ the map
    $$
    y \colon x \mapsto (1 - \eta) \operatorname{sg}(x) + \eta x
        \,, $$
    evaluates to $x$, but has a Jaconbian matrix $\eta I$.
    """

    def __init__(self, *, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return scale_grad(input, self.scale)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


def trailing_lerp(x0, x1, *, eta, leading=1):
    """Unsqueeze the trailing dims of `eta` for linear interpolation.

    See `torch.lerp`: torch.lerp(x0, x1, eta) = (1 - eta) * x0 + eta * x1
    """
    # XXX `x0.lerp(x1, eta) = lerp(x0, x1, eta)` broadcasts correctly
    trailing = (1,) * max(x0.ndim - eta.ndim - leading, 0)
    return x0.lerp(x1, eta.reshape(*eta.shape, *trailing))


def masked_rnn(core, input, hx=None, *, reset=None, h0=None):
    """Apply a rnn layer stack to the sequential input with restarts.

    Parameters
    ----------
    core : torch.nn.Module
        The recurrent core with `batch_first=False`, that accepts an tensor
        `input` with leading sequence and batch dims (`T x B x ...`) and
        the recurrent state `hx` (a `? x B x ...` tensor or container thereof
        or `None`).

        The core must output two objects: the output tensor (`T x B x ...`) and
        the updated state `hx` (nested container of `? x B x ...` tensors or
        `None`). If the core USES a recurrent state, then it MUST always return
        a container. Otherwise, effectively non-recurrent cores MUST always
        return `hx = None`, regardless of the input argument.

        See `torch.nn.modules.rnn`.

    input : torch.Tensor
        The tensor of shape `T x B x ...` with a batch of input sequences.

    hx : torch.Tensor or container of torch.Tensor
        The recurrent state at the start of the given fragment of the input
        sequence. Its tensors must have shape `? x B x ...`. If None, then
        `hx` is set to `h0`.
        See `torch.nn.modules.rnn`.

    reset : torch.Tensor, dtype=bool
        A boolean mask that flags the corresponding item in the input sequence
        of batches as the START of new sub-sequence. This means that the state
        `hx` should be RESET to its initial value BEFORE processing the item.
        If `None`, it is assumed that the sequence never resets.

    h0 : torch.Tensor or container of torch.Tensor
        The initial recurrent state, used to jumpstart recurrent core at the
        very BEGINNING of the whole sequence. If None, then the initial state
        is expected to be initialized to zeros by the core itself.

    Returns
    -------
    output : torch.Tensor
        The output sequence resulting from applying the core to the input
        and the starting recurrent state (`hx` or `h0`).

    hx : torch.Tensor or container of torch.Tensor
        The final recurrent state after running through the input sequence.

    Details
    -------
    Applies the core sequentially over the first dim of `input`, diffably
    resetting the runtime state `hx` to _zero_ or a supplied init `h0`,
    according to the provided sequence restart indicator `reset`.

    If the core DOES NOT use a recurrent state, it is still expected to take
    an `hx` keyword argument and return an update for `hx`, which in this case
    MUST always be `None`.
    """
    # `input` is `T x B x ...`, `reset` is `T x B`
    # `hx` is `? x B x ...`, and `h0` is `? x 1 x ...`
    n_seq, n_batch = input.shape[:2]

    # ignore absent or all-false reset masks
    if reset is None or not reset.any() or n_seq < 1:
        # Use the initial runtime state `h0`, repeated along the batch dim, if
        # `hx` hasn't been supplied, since the `hx` passed to `core()` must
        # have correct batch dims.
        if h0 is not None and hx is None:
            # `h0 <<-- torch.cat([h0, h0, ..., h0], dim=1)`
            # XXX we'd better check if the batch dim of h0 is unit...
            hx = plyr.tuply(torch.cat, *(h0,) * n_batch, dim=1)
            # XXX We do not use `.tile` or `.repeat`, because they require
            # their replication specs to be broadcastible from the trailing
            # dims, thereby constraining dimensionality of `hx` and `h0`.
        return core(input, hx=hx)

    # check the leading dims (seq x batch)
    if reset.shape != input.shape[:2]:
        raise ValueError(f"Dim mismatch {reset.shape} != {input.shape[:2]}")

    # make sure the reset mask is numeric 0-1 for lerp-ing
    reset = reset.to(input)
    if hx is not None:
        # create `h0` as broadcastible scalar zeros from the given `hx`
        if h0 is None:
            h0 = plyr.suply(lambda t: t.new_zeros(()), hx)

        # XXX `hx` could have unit batch, since .lerp broadcasts across `eta`
        hx = plyr.apply(trailing_lerp, hx, h0, eta=reset[0])

    elif h0 is not None:  # `hx` is None!
        # we can safely ignore the first reset, since here `hx` should be `h0`
        hx = plyr.tuply(torch.cat, *(h0,) * n_batch, dim=1)
    # XXX at this point `h0` and `hx` are either both None or both not None.

    # make one step thru the core in order to get the correct recurrent state
    # if `hx` is None on input, then the core must either init it properly, or
    # return None. However, switching from non None to None is forbidden.
    first, hx_ = core(input[:1], hx=hx)
    if hx_ is None and hx is not None:
        raise RuntimeError(
            "`hx` and `h0` must NOT be specified " " if the `core` is non-recurrent."
        )

    # fast branch for one-element sequences
    if n_seq == 1:
        return first, hx_

    # fall back to non-resetting logic, if the first step returns `hx=None`
    if hx_ is None:
        # step through all the remaining input seq elements all at once,
        # since there is no recurrent state to maintain and reset.
        remaining, _ = core(input[1:], hx=None)
        return plyr.tuply(torch.cat, first, remaining, dim=0), None

    # the first core step yielded a meaningful non None `hx`, but the original
    #  `hx` was None, so init `h0` to zeros (default), if it was omitted.
    if h0 is None and hx is None:
        h0 = plyr.suply(lambda t: t.new_zeros(()), hx_)
    hx = hx_

    # loop along `n_seq` dim in slices with fake T=1, diffably resetting
    # `hx` to `h0' when `reset` tells us to by lerp-ing.
    outputs = [first]
    for x, f in zip(input.unsqueeze(1)[1:], reset[1:]):  # XXX does `.unbind`
        # `hx <<-- (1 - r) * hx + r * h0` is `h0 if reset else hx`
        # XXX `hx is not h0` is a bad design choice, since we could have been
        # called with  hx = h0 != None.
        out, hx = core(x, hx=plyr.apply(trailing_lerp, hx, h0, eta=f))
        outputs.append(out)

    return plyr.tuply(torch.cat, *outputs, dim=0), hx


def latched_masked_rnn(
    core,
    input,
    hx=None,
    yx=None,
    *,
    reset=None,
    h0=None,
    latch=None,
):
    # resort to the masked rnn implementation if latch is omitted
    if latch is None or not latch.any():
        return masked_rnn(core, input, hx, reset=reset, h0=h0)

    # Although there is no need to reset the recurrent states, we still have to
    # latch them and the outputs. For this it is simpler to piggy back off the
    # existing loop rather than implement a separate one.
    if reset is None:
        reset = torch.zeros_like(latch)

    # check the leading dims (seq x batch)
    if not (latch.shape == reset.shape == input.shape[:2]):
        raise ValueError("Dim mismatch in `reset`, `latch` and `input`.")

    # make sure the masks are numeric 0-1 for lerp-ing
    reset = reset.to(input)

    # prepare `hx`: if `hx` isn't None, then ensure correct `h0`, otherwise
    # broadcast `h0` along its unit batch dim, PROVIDED it is not omitted.
    n_seq, n_batch = input.shape[:2]
    if hx is not None:
        if h0 is None:
            h0 = plyr.suply(lambda t: t.new_zeros(()), hx)
        hx_ = plyr.apply(trailing_lerp, hx, h0, eta=reset[0])

    elif h0 is not None:
        hx_ = plyr.tuply(torch.cat, *(h0,) * n_batch, dim=1)

    else:
        # if we can not meaningfully init `hx_`, since both `hx`
        #  and `h0` are None, then delegate this to the core.
        hx_ = None

    # compute one step of the core to check the recurrent state
    # XXX at this point `h0` and `hx_` are either both None or both not None
    yx_, hx_ = out_ = core(input[:1], hx=hx_)
    if hx_ is None and (hx is not None or h0 is not None):
        raise RuntimeError(
            "`hx` and `h0` must NOT be specified " " if the `core` is non-recurrent."
        )
    #    hx         h0      =>       hx_         h0     hx
    #     None       None   =>   None           None   None
    #     None   not None   =>   (h0,) * B       --    None
    # not None       None   =>   lerp(hx, 0)    zero    --
    # not None   not None   =>   lerp(hx, h0)    --     --

    # the past output was omitted, init it to zeros
    if yx is None:
        yx = plyr.suply(lambda t: t.new_zeros(()), yx_)

    # the core is non-recurrent and the reset signal is irrelevant
    latch = latch.to(input)
    if hx_ is None:
        # fast branch for one-element sequences
        yx = plyr.apply(trailing_lerp, yx_, yx, eta=latch[0])
        if n_seq == 1:
            return yx, None

        # we compute all outputs at once and latch them in post processing
        outputs = [yx]
        remaining, _ = core(input[1:], hx=None)
        for yx_, s in zip(remaining.unsqueeze(1), latch[1:]):
            yx = plyr.apply(trailing_lerp, yx_, yx, eta=s)
            outputs.append(yx)

        return plyr.tuply(torch.cat, *outputs, dim=0), None

    # either latch to the original hx or the default h0, initted to zero from
    # the meaningful `hx` yielded by the first core step
    if h0 is None:  # fires only when `hx` was omitted
        h0 = plyr.suply(lambda t: t.new_zeros(()), hx_)
    hx = h0 if hx is None else hx

    # if the original hx was omitted, then latch on to the inital h0 instead
    yx, hx = out = plyr.apply(trailing_lerp, out_, (yx, hx), eta=latch[0])
    # XXX use explicit `yx` and `hx`, but also refer to them via `out`
    if n_seq == 1:
        return out  # fast branch for one-element sequences

    # `yx` latched output, `hx` current recurrent state (`h0` default)
    outputs = [yx]
    for x, s, r in zip(input.unsqueeze(1)[1:], latch[1:], reset[1:]):
        # pre-reset the recurrent state `hx` to `h0` if instructed to by
        #  `reset` and compute the output and the recurrent state update
        #     $\hat{h}_t = (1 - r_t) \odot h_t + r_t \odot h_0$
        #     $\hat{y}_t, \hat{h}_{t+1} = g(x_t, \hat{h}_t)$
        out_ = core(x, hx=plyr.apply(trailing_lerp, hx, h0, eta=r))

        # latch the output `yx` and the recurrent state `hx`
        #     $h_{t+1} = (1 - s_t) \odot \hat{h}_{t+1} + s_t \odot h_t$
        #     $y_t = (1 - s_t) \odot \hat{y}_t + s_t \odot y_{t-1}$
        yx, hx = out = plyr.apply(trailing_lerp, out_, out, eta=s)
        outputs.append(yx)  # accumulate the output seq
        # XXX r_t -- "reset recurrent" flag, s_t - 'latch output' flag

    return plyr.tuply(torch.cat, *outputs, dim=0), hx


@torch.no_grad()
def rnn_reset_bias(mod):
    """Open certain rnn gates by positively biasing them."""
    if not isinstance(mod, RNNBase):
        return

    # although classes derived from RNNBase have the same parameter
    #  structure, we still do some limited sanity checking
    for nom, par in mod.named_parameters(recurse=False):
        if not nom.startswith("bias_"):
            continue

        bias = par.unflatten(0, (-1, mod.hidden_size))
        # torch's `nn.LSTM` and `nn.RNN` have redundant biases, but `nn.GRU`
        # does not. In it the `ir-hr` and `iz-hz` bias pairs are redundant,
        # but NOT the `in-hn` pair, since `hn` gets modulated by `r_t` when
        # computing `n_t`.
        #   See help(nn.GRU)
        # Nevertheless, we init b_{h*} `bias_hh_l[k]` to zero and tweak
        #  the initial b_{i*} `bias_ih_l[k]`, depending on the arch.
        if nom.startswith("bias_hh_l"):
            init.zeros_(bias)  # XXX GRU might need special care!!
            continue

        # RNNBase as of 2021-12 has two kinds of bias terms
        if not nom.startswith("bias_ih_l"):
            raise TypeError(f"Unrecognized bias term `{nom}` " f"in `{type(mod)}`.")

        if isinstance(mod, LSTM):
            # bias the forget gates towards open, so that initial
            #  grads through the cell state `c_t` pass uninhibited
            # XXX `nn.LSTM` docs say: bias is [b_ii, b_if, b_ig, b_io]
            # XXX `\sigma(2) \approx 0.88` should be ok
            init.constant_(bias[1], 2.0)

        elif isinstance(mod, GRU):
            # slightly bias the update gates towards open
            # XXX `nn.GRU` docs say: bias is [b_ir, b_iz, b_in]
            # XXX `\sigma(.8) \approx 0.69` seems nice
            init.constant_(bias[1], 0.8)

        elif isinstance(mod, RNN):
            # there isn't anything more we could do for Elman RNN, but make
            #  its outputs initially wobble around zero.
            init.zeros_(bias)

        else:
            raise TypeError(f"Unrecognized recurrent layer `{type(mod)}`.")


@torch.no_grad()
def rnn_reset_weight_hh_ortho(mod, *, gain=1.0):
    """Make unitary hidden-to-hidden transforms."""
    # classes derived from RNNBase have the same parameter structure
    if not isinstance(mod, RNNBase):
        return

    for nom, par in mod.named_parameters(recurse=False):
        if not nom.startswith("weight_hh_"):
            continue

        # init each block as a random orthonormal matrix
        for blk in par.unflatten(0, (-1, mod.hidden_size)):
            init.orthogonal_(blk, gain=gain)


@torch.no_grad()
def rnn_reset_weight_hh_eye(mod, *, gain=1.0):
    """Make near-identity hidden-to-hidden transforms."""
    # TODO figure out the correct fan-in and motivation
    pass

    # classes derived from RNNBase have the same parameter structure
    if not isinstance(mod, RNNBase):
        return

    for nom, par in mod.named_parameters(recurse=False):
        if not nom.startswith("weight_hh_"):
            continue

        # init each block as a random orthonormal matrix
        for blk in par.unflatten(0, (-1, mod.hidden_size)):
            # allow small leakage from the other hiddens
            blk.normal_(0.0, gain / math.sqrt(mod.hidden_size))

            # ..., while letting own value propagate unmodulated
            blk.diagonal()[:] = 1.0


def rnn_hx_shape(self):
    """Get the batch-broadcastible shape of the recurrent state.

    Complaint
    ---------
    Torch devs should've made this into a recurrent module's method.
    """
    if not isinstance(self, (RNNBase, GRU, LSTM)):
        raise TypeError(f"Unrecognized recurrent layer `{type(self)}`.")

    n_hidden_states = self.num_layers * (2 if self.bidirectional else 1)

    # order matters due to possible subclassing!
    if isinstance(self, GRU):
        # copied almost verbatim from `torch.nn.GRU.forward`
        return torch.Size((n_hidden_states, 1, self.hidden_size))

    elif isinstance(self, LSTM):
        # copied almost verbatim from `torch.nn.LSTM.forward`
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size

        return (
            torch.Size((n_hidden_states, 1, real_hidden_size)),  # h
            torch.Size((n_hidden_states, 1, self.hidden_size)),  # c
        )

    elif isinstance(self, RNNBase):
        # copied almost verbatim from `torch.nn.RNNBase.forward`
        return torch.Size((n_hidden_states, 1, self.hidden_size))

    # not reached


def multinomial(
    input,
    /,
    num_samples=1,
    dim=-1,
    *,
    replacement=False,
    generator=None,
    squeeze=True,
):
    """Draw `num_samples` random integers from the unnormalized multinomial
    distribution, located in the specified `dim` of the `input` tensor.

    Details
    -------
    Unlike, `torch.multinomial`, this version SUPPORTS multidimensional input,
    but DOES NOT allow pre-allocated `out` storage.

    The returned integer-valued tensor MIGHT NOT be contiguous in memory due
    to dimension permutations.
    """
    # compute the split index for rolling the dims
    dims = list(range(input.ndim))
    _, dim = divmod(dim, len(dims))
    pos = dim + 1

    # roll proba dim to trailing position
    per = input.permute(dims[pos:] + dims[:pos])

    # sample the multinomial, placing the variates at the last dim
    out = torch.multinomial(
        per.flatten(0, -2),
        num_samples,
        replacement=replacement,
        generator=generator,
        out=None,
    ).unflatten(0, per.shape[:-1])

    # undo dim rolling and optionally squeeze the output
    out = out.permute(dims[-pos:] + dims[:-pos])
    return out.squeeze(dim) if squeeze else out


def apply_mask(tensor, mask, *, value=-float("inf")):
    """Mask the values in the given tensor."""

    return tensor.masked_fill(mask.to(bool), value)


def masked_softmax(raw, mask, dim=-1, *, value=None):
    """Compute the probabilites from the unnormalized logits `raw` along
    the indicated `dim`, optionally masking using `mask`.
    """
    # '-inf' masking should be applied to the unnormalized logits
    if mask is not None:
        raw = apply_mask(raw, mask, value=value)

    # sample from the masked distribution
    return raw.softmax(dim=dim)


def masked_multinomial(raw, mask, dim=-1, *, value=None):
    """Draw a variate from the categorical rv, specified by the unnormalized
    logits `raw` at the indicated `dim`, optionally masked by `mask` boolean
    array of the same shape as `raw`.
    """
    proba = masked_softmax(raw.detach(), mask, dim=dim, value=value)

    # sample from the masked distribution
    return multinomial(proba, 1, dim).squeeze(dim)


class ParameterContainer(Module):
    """Module-compatible nested container of parameters, which keeps references
    to the leaf parameters.

    Details
    -------
    Torch's stock `nn.ParameterList` or `nn.ParameterDict` duck-type list and
    dict containers, respectively, by keeping and registering the contained
    parameters directly in `._parameters` odict of `nn.Module`, which they are
    subclasses. However, they do not provide full support for nested containers
    of parameters.

    Example
    -------
    >>> import troch, plyr
    >>> from collections import namedtuple
    >>> NT = namedtuple('NT', 'u,v')
    >>> sz = torch.Size((7, 3, 5))
    >>> struct = (sz, sz, {'a': sz, 'z': NT(sz, sz)},)
    >>> h0 = plyr.apply(nn.Parameter, plyr.apply(torch.randn, struct))
    >>> pc = ParameterContainer(h0)
    >>> pc.half();
    >>> pc._value()
    """

    def __new__(cls, parameters):
        # regress to nn.Parameter if non-container
        if isinstance(parameters, Parameter):
            # keep a weak ref to self
            if not hasattr(parameters, "_value"):
                parameters._value = weakref.ref(parameters)

            return parameters

        return object.__new__(cls)

    def __init__(self, parameters):
        # We bypass nn.Module's parameter registration logic by regressing
        # to `object`. Any device move or dtype change caused by nn.Module
        # is made on parameters IN-PLACE, and thus is reflected in the original
        # built-in containers, since they store data by reference.
        object.__setattr__(self, "_value", lambda: parameters)
        # XXX This does not create cyclic reference, and is just an alternative
        # container of the parameters.

        # Here we let nn.Module's smart __setattr__ to properly register
        # parameters and containers thereof: if the item in the container is
        # an `nn.Parameter`, then the __new__ of this class (above, called via
        # `type(self)(it)`) will return it intact and nn.Module will register
        # it in its `._parameters` odict. Otherwise, __new__ creates this class
        # (a subclass of nn.Module), which is registered in `_modules` odict.
        super().__init__()

        if isinstance(parameters, tuple) and hasattr(parameters, "_fields"):
            parameters = parameters._asdict()

        if isinstance(parameters, (tuple, list)):
            for j, it in enumerate(parameters):
                setattr(self, str(j), type(self)(it))

        elif isinstance(parameters, dict):
            for k, it in parameters.items():
                setattr(self, k, type(self)(it))

        else:
            raise TypeError(f"Unsupported `{type(parameters).__name__}`.")

    def __getitem__(self, key):
        # emulate container item access
        return getattr(self, str(key))

    def __setitem__(self, key, value):
        raise IndexError(key)

    def extra_repr(self):
        # use ParameterList's extra_repr implementation
        tmpstr = ParameterList.extra_repr(self)
        return "\n".join(map(str.strip, tmpstr.splitlines()))
