import torch
import numpy as np

from copy import deepcopy
from plyr import apply, suply, iapply

from .engine import Input


def launch(capsule, initial):
    """Launch the freshly created capsule."""
    assert isinstance(initial, Input)

    # (capsule) start with the handshake
    if capsule.send(None) is not None:
        raise RuntimeError("Capsule handshake failed.")

    # (capsule) communicate the initial input and gets the reaction to it
    return capsule.send(initial)


def capsule(step, update, length, *, device=None):
    """T-BPTT trajectory collector for capsuled RL. See docs `.engine.step`.

    Parameters
    ----------
    step: callable
        A function taking `input` (namedtuple with fields `obs`, `act`, `rew`,
        and `fin`, which represents the recent observations) and the keyword
        `hx` (arbitrarily nested tensors), which stores the recurrent runtime
        state, that may be auto-initialised if `hx` is None. `step` returns
        the action `act`, auxiliary data `output`, and the new state `hx`.

    update: callable
        A function that updates whatever internal parameters `step` depends on,
        and takes in `input` (always non-diffable `obs`, `act`, `rew`, `fin`),
        `output`, `hxx` the recent history of `hx` states, possibly diff-able,
        and `nfo` history of info data. It returns an information dict, and an
        update for `hx` (may be diff-able) to be used at the start of the next
        fragment, or `None` in which case the current inner value is kept.

    length: int
        The length of trajectory fragments used for each truncated bptt grad
        update.
    """
    if update is None and length >= 1 or update is not None and length < 1:
        raise ValueError("`update` can be None iff fragment `length` is zero.")

    # (capsule) the tensor cloning func, since host-device moves produce a copy
    device = torch.device("cpu") if device is None else device
    cloner = torch.clone if device.type == "cpu" else lambda t: t.to(device)
    # XXX `.to` is enough here as the npy/pyt buffers are "on host" by design

    # (capsule) finish handshake and prepare the npyt state (aliased npy-pyt)
    # XXX no need to create `AliasedNPYT`, since we live in a capsule!
    pyt = suply(torch.as_tensor, suply(np.copy, Input(*(yield None))))
    if device.type != "cpu":
        pyt = suply(torch.Tensor.pin_memory, pyt)
    npy = suply(np.asarray, pyt)  # XXX `npy` aliases `pyt` (thru array proto)
    suply(torch.Tensor.unsqueeze_, pyt, dim=0)  # fake seq dim

    # (sys) collect trajectory in fragments, when instructed to
    fragment = []
    append = id if length < 1 else fragment.append  # `id` serves as a dummy
    length = max(length, 1)  # XXX clamp the length to one anyway

    # let the learner properly init `hx`-s batch dims
    # XXX `hx` is current, `gx` is at the start of the fragment
    gx = hx = None  # XXX `hx` is either None, or an object

    # (sys) perpetual rollout
    nfo_, hxx = None, [gx]  # XXX the initial info dict is unavailable
    while True:  # .learn()
        # (sys) clone for diff-ability, because `pyt` is updated in-place
        input = suply(cloner, pyt)
        nfo = nfo_

        # REACT and update runtime recurrent state from t to t+1
        #   x_t, a_{t-1}, h_t -->> a_t, y_t, h_{t+1}
        #   with `a_t \sim \pi_t`
        #  XXX if the runtime state is irrelevant to `step`, then it returns
        #  `hx` intact
        act_, output, hx = step(input, hx=hx, nfo=nfo)

        # (sys) update the action in `npy` through `pyt`
        suply(torch.Tensor.copy_, pyt.act, act_)

        # STEP and advance local time
        #   \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}, I_{t+1}
        #   with \omega_t being the unobservable complete state, and I_{t+1}
        #   -- the dict with information relevant to the t -->> t+1 transition.
        obs_, rew_, fin_, nfo_ = yield npy.act
        # XXX adding a skip logic here is dumb: just don't `.send` anything
        #  into this capsule!

        # (sys) update the rest of the `npy-pyt` aliased context
        suply(np.copyto, npy.obs, obs_)
        suply(np.copyto, npy.rew, rew_)  # XXX allow structured rewards
        np.copyto(npy.fin, fin_)  # XXX must be a boolean scalar/vector

        # (sys) we deepcopy the info dict `nfo_`, but report on the NEXT step
        # XXX we ASSUME `nfo_` $I_{t+1}$ is a dict, but do not test it
        nfo_ = deepcopy(nfo_)  # XXX deepcopy in case `nfo` reuses its buffers

        # (sys) collect a fragment of time `t` afterstates t=0..N-1
        # XXX `input` and `nfo` are SIMULTANEOUS, unlike `engine.step`!
        append(((input, output), nfo_ if nfo is None else nfo))
        if hx is not None:
            hxx.append(hx)

        if len(fragment) < length:
            continue

        # (sys) one-step look-ahead $y_N$, e.g. value-to-go bootstrap.
        # DO NOT yield action to the caller, nor update `npy-pyt`, nor `hx`!
        input = suply(cloner, pyt)
        _, output, _ = step(input, hx=hx)
        append(((input, output), nfo_))

        # (sys) repack data ((x_t, a_{t-1}, r_t, d_t), y_t), t=0..N
        # XXX note, `.act[t]` is $a_{t-1}$, but the other `*[t]` are $*_t$,
        #  e.g. `.rew[t]` is $r_t$, and `output[t]` is `$y_t$
        # XXX `nfo` is the auxiliary data associated with each step in input
        # is not collated, since the info dict is entirely env-dependent.
        chunk, nfo = zip(*fragment)
        input, output = apply(torch.cat, *chunk, _star=False)  # dim=0
        fragment.clear()

        # (sys) do an update on the collected fragment and get the revised
        # recurrent runtime state for the next fragment
        _, gx = update(input, output, hxx=hxx, nfo=nfo)
        hx = gx = hx if gx is None else gx  # if None, set gx to hx
        hxx = [gx]


def buffered(step, process, length, *, device=None):
    """Buffered non-diffable trajectory collector for capsuled RL.

    Parameters
    ----------
    step: callable
        A function taking `input` (namedtuple with fields `obs`, `act`, `rew`,
        and `fin`, which represents the recent observations), the keyword `hx`
        (arbitrarily nested tensors), which stores the recurrent runtime state,
        that may be auto-initialised if `hx` is None, and the most recent info
        data `nfo`. `step` returns the action `act`, auxiliary data `output`,
        and the new state `hx`.

    process: callable
        A function that updates whatever internal parameters `step` depends on,
        and takes in `input` (always non-diffable `obs`, `act`, `rew`, `fin`),
        `output`, `hxx` the recent history of `hx` states, possibly diff-able,
        and `nfo` history of info data. It returns an information dict, and an
        update for `hx` (may be diff-able) to be used at the start of the next
        fragment, or `None` in which case the current inner value is kept.

    length: int
        The length of trajectory fragments used for each truncated BPTT grad
        update.

    Details
    -------
    `yield` statement governs the LOCAL time tick of this capsule.

    `obs`, `act`, `rew` and `fin` are STRONGLY structured: this data has static
    higher-level structure and unchanging lower-level shape and dtype. Thus it
    is guaranteed to be collated and sent to the requested device.

    `nfo` on the other hand is WEAKLY structured, i.e. we can be only sure that
    it is a container of dynamically changing content. Therefore it is relayed
    to `step` and `process` as is.

    We non-diffably track the output of `step()` by stacking it results in
    order to facilitate `collect-process-consume` trajectory pipeline design.
    This incurs an extra call to `step()` for value bootstrap and might have
    a side effect on PRNG, if `step()` relies on it.
    """
    assert length >= 1
    device = torch.device("cpu") if device is None else device

    # ensure numpy arrays and then make torch tensor COPIES
    pyt = suply(torch.as_tensor, suply(np.copy, Input(*(yield None))))
    if device.type != "cpu":
        pyt = suply(torch.Tensor.pin_memory, pyt)

    # alias torch's storage with numpy arrays and add a fake leading dim
    npy = suply(np.asarray, pyt)
    input = pyt = suply(torch.Tensor.unsqueeze_, pyt, dim=0)
    if device.type != "cpu":
        input = suply(lambda t: t.to(device), pyt)
        # XXX `pyt` and `input` diverge here: `pyt` resides in the pinned
        #  memory on the host, whereas `input` has been copied to device.

    # allocate buffers on the correct device and get its single-step
    # editable slices. `+ 1` is the one-step ahead overlap
    # XXX Since the number of chunks equals the size along dim 0, `torch.chunk`
    #  returns a tuple of views into the input with unit size dim 0, unlike
    #  `.unbind`, which drops the sliced dim.
    buffer = apply(torch.cat, *(input,) * (length + 1), _star=False, dim=0)
    vw_buffer = iapply(torch.chunk, buffer, chunks=length + 1, dim=0)

    # `step(...)` should properly init `hx`-s batch dims
    # XXX runtime state meanings: 'gx` before fragment, `hx` current
    gx = hx = None  # XXX `hx` is either None, or an object

    # perpetual rollout
    nfo_, hxx = None, [gx]  # XXX the initial info dict is unavailable
    nfos, outs = [], []
    while True:  # .learn()
        # write the current `pyt` into the current slice of the buffer
        # XXX `input` is structurally IDENTICAL to `vw_buffer` and refers to
        #  the SAME tensors, since `d.copy_(s)` copies INPLACE and returns `d`.
        input = suply(torch.Tensor.copy_, vw_buffer[len(nfos)], pyt)
        nfo = nfo_

        # REACT and update the action in `npy` through `pyt`
        #   x_t, a_{t-1}, h_t -->> a_t, y_t, h_{t+1}  with `a_t \sim \pi_t`.
        with torch.no_grad():  # XXX `yield` NEVER causes `__exit__`!
            act_, out_, hx = step(input, hx=hx, nfo=nfo)
            suply(torch.Tensor.copy_, pyt.act, act_)

        # STEP and advance local time
        #   \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r_{t+1}, d_{t+1}, I_{t+1}
        #   with \omega_t being the unobservable complete state of the ENV
        obs_, rew_, fin_, nfo_ = yield npy.act

        # update the rest of the `npy-pyt` aliased context
        suply(np.copyto, npy.obs, obs_)
        suply(np.copyto, npy.rew, rew_)
        np.copyto(npy.fin, fin_)

        # we save the output y_t and the new runtime state h_{t+1}. As for the
        #  new `nfo_`, we make its deep copy to keep it for the next step, but
        #  save the previous `nfo`, which is in sync with $z_t$ in `input` (NOT
        # `pyt` which has been updated to $z_{t+1}).
        nfo_ = deepcopy(nfo_)
        nfos.append(nfo_ if nfo is None else nfo)
        outs.append(out_)
        if hx is not None:
            hxx.append(hx)

        if len(nfos) < length:
            continue  # XXX loop-and-a-half?

        # write the last `pyt` into the buffer and get bootstrap output
        input = suply(torch.Tensor.copy_, vw_buffer[len(nfos)], pyt)
        with torch.no_grad():
            _, out_, _ = step(input, hx=hx, nfo=nfo_)

        # collate info dicts
        nfo = *nfos, nfo_
        nfos.clear()

        # joint the output recirs into a single buffer
        # XXX In order to implement hot-swappable buffer, call `step()` once,
        #  when initializing `vw_buffer`
        output = apply(torch.cat, *outs, out_, _star=False)  # implied dim = 0
        outs.clear()

        # process the collected fragment and revise the recurrent runtime state
        #  for the next fragment (runtimes: `gx, hx = hxx[0], hxx[-1]`)
        _, gx = process(buffer, output, hxx=hxx, nfo=nfo)
        hx = gx = hx if gx is None else gx  # if None, set gx to hx
        hxx = [gx]
