import plyr
import numpy as np
import torch
from itertools import chain


def extract_truncated(input, hxx=None, nfo=None):
    r"""Collect input-runtime data from truncated episodes for value-to-go correction.

    Implementation
    --------------
    The procedure ONLY EXTRACTS the relevant data from the recorded fragment, but
    NEVER modifies it. It expects the `input` to be a `.engine.Input` namedtuple
    of T x B x ... PYTORCH TENSORS storing `B` trajectory fragments of length `T`.
    The `hxx` argument is expected to be TENSORS with recurrent runtime states,
    the `dim=1` of which is the BATCH dimentsion of size `B`.

    Generally, `nfo` is a tuple of tupels of dict, containing arbitrary objects,
    however this implementation relies on the info-dict update logic upon `truncation`
    in [SerialVecEnv](nle_toolbox/utils/rl/engine.py#L61-68). Specifically, the
    wrapper saves the new observation under `obs` key in `nfo` dict when it sees
    a raised `TimeLimit.truncated` flag. The `obs` data in the inner dict of `nfo`
    is expected to be a numpy array and the key itself is expected to be PRESENT
    in those dicts of the nested `nfo` tuple, for whic the corresponding flag in
    `input.fin` is `-1`. If the the data in `input.fin` is boolean, then this
    procedure is bypassed.

    Details
    -------
    An episode may end either due to _termination_ or _truncation_. The former
    occurs when the environment has reached a true terminal state from which no
    state transition can happen and after which all rewards are ZERO. The latter
    takes place when the environment per se reached a non-terminal state, yet
    the for some reason the environment's simulator signals that the episode has
    finished. This detail affects the logic of multistep lookahead value-to-go
    estimates and the derived quantities (such as advantages and TD residuals).
    The value backups over rollouts ending with a terminal state use zero terminal
    value approximation, whereas for truncated rollouts one should use the value
    at the last state.

    This can be achieved by using adjusted value-rewards, which affect only at the
    reward receivable for getting to the truncated state: $
        r^\circ_t = r_t + \gamma_t v_t 1_{f_t = -1}
    $, where the reward for _transitionion to a state_ is $r_t$, the termination
    flag is $f_t \in \{-1, 0, +1\}$, and the value-to-go _starting from that state_
    is $v_t$. For this to work is is necessary to recompute the value for the
    original states the truncated episodes ended up in. For the auto-resetting
    vectorized environments this means that the last observation before the reset
    has to be used. (see `gamma()` in `.returns`). Note, that the simplest solution
    is to correct the rewards, rather than overwriting the value estimates, since
    the latter breaks the td-residuals of the initial observation.

    Without this extra step, due to the way the trajectory data is recorded by
    the experience collector, the reward gets corrected by the value-to-go of the
    episode's initial observation, thereby virtually looping the reward stream
    via bootstrapping. It has been observed that during training the agent may
    enter a transient phase, wherein it discovers a well-performing policy and
    eventually backs its value estimate up to the initial state. Then, upon stably
    learning a sufficiently high value-to-go estimate at the initial state, it
    becomes more lucrative for the agent to intentionally truncate episodes, since
    in this case it would get rewarded with its own value-to-go.
    """
    # get the mask of truncated steps
    # XXX we ignore zero-th record since the fragments always overlap, i.e.
    #  the T-th record IS the 0-th record of the next fragment.
    # XXX `.lt(0)` preserves compatibility with boolean termination flags
    mask = input.fin.lt(0)
    mask[0].zero_()
    if not mask.any():
        return None, None, None

    # (sys) copy the post truncation inputs and mark them as non-terminal
    # XXX z_t = (x'_0, a_{t-1}, r_t, f_t) with f_t = -1, reward r_t for action
    #  a_{t-1} (t-1 -->> t), and x'_0 the observation  emitted by the inital
    #  state of a reset env. `mask` selects such t, that have `f_t = -1`.
    trunc = plyr.apply(lambda x: x[mask].unsqueeze(0), input)
    trunc.fin.zero_()

    # (sys) recover the original observations x_t, overwritten by auto-resets
    # XXX the numpy arrays are stacked and converted to torch tensors
    # XXX flattened mask and `tuple-chain` is faster than an array of object
    flatmask = mask.flatten().nonzero()[:, 0].tolist()
    nfo_ = tuple(map(tuple(chain(*nfo)).__getitem__, flatmask))
    obs_ = plyr.apply(np.stack, *[n["obs"] for n in nfo_], _star=False)
    obs_ = plyr.suply(torch.as_tensor, obs_)

    # (sys) graft the original observations into the extracted inputs
    plyr.suply(torch.Tensor.copy_, trunc.obs, obs_)

    # (sys) return if the rollout data is has no recurrent runtime state
    if not isinstance(hxx, (list, tuple)) or hxx[-1] is None:
        return mask, trunc, None

    # (sys) fetch the recurrent states from `hxx`
    # XXX although we assume the recurrent states are torch tensors, it is
    #  debatable whether initting to zeros is a good idea. However hxx[0]
    #  is never picked by design, hence may be arbitrary.
    hx0, *hxx_ = hxx
    if hx0 is None:
        hx0 = plyr.apply(torch.zeros_like, hxx_[-1])

    # (sys) get a C-order flattened tuple of runtimes unbound along the batch dim,
    # then select according to mask's nonzeros and stack along the same dim.
    hx_ = tuple(chain(*zip(*plyr.iapply(torch.unbind, (hx0, *hxx_), dim=1))))
    hx = plyr.apply(torch.stack, *map(hx_.__getitem__, flatmask), _star=False, dim=1)

    # (sys) return the mask and the patched inputs needed for recomputing
    return mask, trunc, hx
