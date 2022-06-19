import torch

from typing import Union
from torch import Tensor


def trailing_broadcast(
    what: Tensor,
    to: Tensor,
) -> Tensor:
    """Add extra trailing unitary dims to `what` for broadcasting it to `to`."""
    return what.reshape(what.shape + (1,) * max(to.ndim - what.ndim, 0))


def gamma(
    rew: Tensor,
    gam: Union[float, Tensor],
    val: Tensor,
    *,
    fin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Prepare the discount factor array and the rewards.

    The discount factor `gam` can be a scalar, or a [T x B x ...] tensor,
    which can be broadcasted to `rew` from its leading dimensions (like `fin`).
    `fin` is always a [T x B] tensor (2-dim) with either of bool or int8 dtype.
    Ternary `fin` is necessary for implementing different value-to-go logic for
    trunctation/termination events: `termination` means zero value estimate,
    `truncation` -- self bootstrapped estimate.
    """
    # align the termination mask with the rewards `rew` by broadcasting from
    #  the leading dimensions
    fin_ = trailing_broadcast(fin, rew)

    # either create a new gamma tensor, or align the given one with `rew`
    if isinstance(gam, float):
        gam_ = rew.new_full(fin_.shape, gam)

    else:
        # make sure to `.clone` gamma since we will be overwriting it soon
        gam_ = trailing_broadcast(gam, rew).clone()

    # Truncation and termination have different effects on the present-value,
    #  despite both blocking the reward flow from the future. Termination means
    #  that the reward flow following the state is zero, which implies that the
    #  value-to-go is zero. Truncation on the other hand means that the episode
    #  has been terminated at a state that may potentially have non-zero future
    #  reward stream. Hence in this case the value-to-go could be bootstrapped
    #  from the current value function. Conceptually, truncation always happens
    #  at the edge of a fragment, when value estimate is used to bootsrap.
    # XXX one-step adjusted rewards r^\circ_t = r_t + \gamma_t v_t 1_{f_t = -1}
    # XXX `rew[t] = r_t`, `fin[t] = f_t`, and `val[t] = v_t` with t = 0..T
    rew_ = torch.addcmul(rew, gam_.masked_fill(fin_.ge(0), 0.0), val)

    # annihilate the discounts according to the truncation/termination mask,
    return fin_, gam_.masked_fill_(fin_.ne(0), 0.0), rew_


def pyt_ret_gae(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    lam: float,
    fin: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute the Generalized Advantage Estimates and the Returns.

    Details
    -------
    The sequential data in `rew` [T x B x ...], `fin` [T x B x ...] and `val`
    [T x B x ...] (and potentially in `gam` [T x B x ...]) is expected to have
    the following timing: `rew[t]` is the reward for the transitioning to a
    state at step $t$, `val[t]` is the estimate of the VALUE-TO-GO STARTING
    FROM this state, `gam[t]` is the discount to apply to this value-to-go
    in order to back it up for past values, and ternary `fin[t]` indicates if
    this state is terminal, truncated, or neither. Formally, the meaning is as
    follows: for the env transition
    $$
      ENV: \omega_{t-1}, a_{t-1}             -->> \omega_t, x_t, r_t, f_t
      ACT: h_{t-1}, (x_t, a_{t-1}, r_t, f_t) -->> h_t, a_t, v_t, \dot{\gamma}_t
        \,, $$

    $r_t$ is `rew[t]`, `fin[t]` is $f_t \in \{0, \pm 1\}$ with `-1` indicating
    truncation, `+1` -- termination and `0` -- continuation, `val[t]` is $v_t$
    and `gam[t]` is $\dot{\gamma}_t$. The factor $\gamma_t$ is the conversion
    rate from `t` to `t-1` reward units, $r_t$ is in `t-1` units since it is
    determined by both endpoints of the `t-1 -->> t` transition, and $v_t$ is
    in `t` units.

    It is important to note that $v_t$ is the value-to-go estimate of the FUTURE
    STREAM of rewards $(r_s)_{s > t}$ which does NOT include $r_t$, since it HAS
    BEEN RECEVIED for `t-1 -->> t`. In terms of the argument names
    `val[t] ~ present_value(rew[t+1:], gam[t+1:])` and $
        v_{t-1} \approx r_t + \gamma_t v_t
    $ -- approximate value of potential transitions from `t-1`.

    If a state is TERMINAL ($f_{t+1} = +1$), then the reward stream coming
    AFTER it is assumed to be zero ($r_s = 0$ for all $s > t$). Hence, the
    returns, time-deltas and bootstrapped estimates are computed using
    $\gamma_t = \dot{\gamma}_t 1_{f_t = 0}$ series instead of plain `gamma`.
    See `gamma()` about truncation events $f_t= - 1$ and the discounts `gam`.
    """

    # get properly broadcasted and zeroed discount coefficients
    fin_, gam_, rew_ = gamma(rew, gam, val, fin=fin)

    # [O(T B F)] compute the td-errors for truncation-aware GAE
    # XXX for variable \gamma_t, the n-step lookahead td-residual over
    #  the current trajectory is
    #  \delta^n_t
    #    = \sum_{j=0}^{n-1} \Gamma^j_{t+1} r_{t+j+1} + \Gamma^n_{t+1} v_{t+n} - v_t
    #    = \sum_{j=0}^{n-1} \Gamma^j_{t+1} \delta_{t+j}  % just expand!
    #  with \delta_t = r_{t+1} + \gamma_{t+1} v_{t+1} - v_t and
    #  \Gamma^j_t = \prod_{k<j} \gamma_{t+k}. For example, in our case
    #    \gamma_t = \tilde{gamma}_t 1_{f_t = 0}, for \tilde{gamma}_t discounts
    delta = torch.addcmul(rew_, gam_, val)[1:].sub_(val[:-1])
    # XXX `bootstrapping` means using $v_{t+h}$ (the func $v(.)$ itself) as an
    # approximation of the present value of future rewards $(r_{j+1})_{j\geq t+h}$

    # rew[t], fin[t], gam[t], val[t] are $r_t$, $f_t$, $\gamma_t$ and $v_t$
    #   `t` is `-j`, `t-1` is `-j-1` (j=1..T)
    # `delta[t]` is `\delta_t` and depends on $(r_s)_{s > t}$
    gae, ret = torch.zeros_like(val), val.clone()
    for j in range(1, len(val)):
        # GAE [O(B F)] A_t = \delta_t + \lambda \gamma_{t+1} A_{t+1} 1_{f_{t+1} = 0}
        # A_t = (1 - \lambda) \sum_{j \geq 0} \lambda^j \delta^{j+1}_t
        #     = \sum_{k \geq 0} \Gamma^k_t \lambda^k \delta_{t+k}
        #     = \delta_t + \lambda \gamma_{t+1} A_{t+1}
        # gae[t] = delta[t] + gam[t+1] * C * (fin[t+1] == 0) * gae[t+1] t=0..T-2
        # XXX `gam_` is already zeroed according to the `fin == 0` mask!
        torch.addcmul(delta[-j], gam_[-j], gae[-j], value=lam, out=gae[-j - 1])

        # RET [O(B F)] G_t = r_{t+1} + \gamma_{t+1} G_{t+1} 1_{f_{t+1} = 0}
        # G_t = \sum_{j\geq 0} \Gamma^j_{t+1} r_{t+j+1}
        #     = r_{t+1} + \gamma_{t+1} G_{t+1}
        # ret[t] = rew[t+1] + gam[t+1] * (fin[t+1] == 0) * ret[t+1] t=0..T-2
        torch.addcmul(rew_[-j], gam_[-j], ret[-j], out=ret[-j - 1])
    # XXX this loop version has slight overhead which is noticeable only for
    # short sequences and small batches, but otherwise this scales linearly
    # in all dimensions. Double buffering for `ret` and `gae` increases the
    # complexity of each iteration by T times, e.g`pyt_multistep_returns`.
    #   `torch.addcmul(rew, gam_, ret[j, 1:], out=ret[1-j, :-1])`

    # `ret[t]` is the truncated total return after state `t` with terminal value
    #  approximation, `gae[t]` is the advantage for `t -->> t+1` transitions
    return ret, gae[:-1]


def pyt_vtrace(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    rho: Tensor,
    fin: Tensor,
    *,
    r_bar: float = float("+inf"),
    c_bar: float = float("+inf"),
) -> Tensor:
    r"""Compute the V-trace state-value estimate.

    Details
    -------
    The notation in this brief description deviates from

        [Espeholt et al. (2018)](http://proceedings.mlr.press/v80/espeholt18a.html)

    The TD(0) residuals are given by
    $$
        \delta_t = r_{t+1} + \gamma_{t+1} 1_{f_{t+1} = 0} v_{t+1} - v_t
        \,, $$

    where $r_t$ (`rew[t]`) is the reward for transitioning to `t` from `t-1`,
    $\gamma_t$ (`gam[t]`) is the discount factor (potentially variable), e.g.
    $\gamma_t = \gamma 1_{f_t = 0}$ for $f_t$ (`fin[t]`), and $v_t$ (`val[t]`)
    is the state-value estimate. In order to support truncation termination we
    use 1-step appxorimation $
        r^\circ_t = r_t + \gamma_t v_t 1_{f_t = -1}
    $ instead of $r_t$.

    The $n$-step lookahead v-trace value estimate is
    $$
        \hat{v}^n_t  % n-step lookahead importance-weighted value estimate
            = v_t + \sum_{j=0}^{n-1} q^j_t \Gamma^j_{t+1} \delta_{t+j}
        \,, $$

    where the action log likelihood ratio $
        \rho_t = \log \pi(a_t \mid z_t) - \log \mu(a_t \mid z_t)
    $ (`rho[t]`) $\mu$ is the behavior policy and $\pi$ is the target, $
        \Gamma^j_t = \prod_{k < j} \gamma_{t+k}
                   % \Gamma^{j+1}_t = \gamma_t \Gamma^j_{t+1}
    $, $
        q^j_t = \biggl( \prod_{k < j} c_{t+k} \biggr) \eta_{t+j}
              % q^{j+1}_t = c_t q^j_{t+1}
    $, $
        \eta_t = \min\{ e^\rho_t, \bar{\eta} \}
    $, $
        c_t = \min\{ e^\rho_t, \bar{c} \}
    $. At $n \to \infty$ limit (bounded $\rho_t$, $v_t$, and $r_t$) we get a
    forward recurrence
    $$
        \hat{v}_t
            % = v_t + \sum_{j \geq 0} q^j_t \Gamma^j_{t+1} \delta_{t+j}
            % = v_t + q^0_t \delta_t
            %   + \sum_{j \geq 0} q^{j+1}_t \Gamma^{j+1}_{t+1} \delta_{t+1+j}
            = v_t + \eta_t \delta_t
              + c_t \gamma_{t+1} (\hat{v}_{t+1} - v_{t+1}) 1_{f_{t+1} = 0}
        \,. $$

    See `gamma()` about the discount factor `gam` and adjusted rewards.
    """
    # get properly broadcasted and zeroed discount coefficients
    fin_, gam_, rew_ = gamma(rew, gam, val, fin=fin)

    # add extra trailing unitary dims for broadcasting to the log-likelihood
    # XXX rho is the current/behavioural likelihood ratio for the taken action
    rho_ = rho.reshape_as(fin_[:-1]).exp()  # NB extra copy by `.exp()`
    eta = rho_.clamp(max=r_bar)

    # [O(T B F)] get the clipped \eta_t-weighted td(0)-residuals \delta_t
    delta = torch.addcmul(rew_, gam_, val)[1:].sub_(val[:-1]).mul_(eta)

    # [O(T B F)] compute $c_t \gamma_{t+1} 1_{f_{t+1} = 0}$
    adv = torch.zeros_like(val)
    seegam = rho_.clamp(max=c_bar).mul_(gam_[1:])
    for j in range(1, len(val)):
        # V-trace [O(B F)]
        #   \hat{v}_t = \hat{a}_t + v_t
        #   \hat{a}_t = \eta_t \delta_t + c_t \gamma_{t+1} 1_{f_{t+1} = 0} \hat{a}_{t+1}
        # adv[t] = rho[t] * delta[t] + gam[t+1] * see[t] * (fin[t+1] == 0) * adv[t+1]
        torch.addcmul(delta[-j], seegam[-j], adv[-j], out=adv[-j - 1])

    vtrace = adv + val  # XXX adv[-1] is zero

    # [O(T B F)] get the importance-weighted 1-step lookahead advantages
    # see sec. "v-trace actor-critic algo" (p. 4) in Espeholt et al. (2018)
    adv = torch.addcmul(rew_, gam_, vtrace)[1:].sub_(val[:-1]).mul_(eta)

    # return the V-trace value targets and the advantages
    return vtrace, adv


def pyt_multistep(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
    *,
    n_lookahead: int = 4,
) -> Tensor:
    r"""Compute the multistep lookahead bootstrapped returns.

    Details
    -------
    see `pyt_ret_gae` about synchronization of `rew`, `fin` and `val`.
    `rew` is on element shorter than `val`! See `gamma()` about `gam`.

    This computes the $l$-step lookahead bootstrapped return (value estimate):
    $$
        G^l_t
            = \sum_{j=0}^{l-1}
                \biggl\{ \prod_{k < j} \gamma_{t+k+1} \biggr\}
                r_{t+k+1}
            + \biggl\{ \prod_{k < l} \gamma_{t+k+1} \biggr\}
                v_{t+l}
                % bootstrapped value-to-go estimate at state t+l. It does
                % not include r_{t+l}, since this reward has been received
                % SIMULTANEOUSLY with transitioning to the state.
        \,, $$

    where $
        \gamma_t = \dot{\gamma}_t 1_{f_t = 0}
    $ -- the observed conditional probability of not terminating on the t-th
    step, $f_t \neq 0$ indicates if step $t$ was final, and
    $$
        v_t \approx
            G^\infty_t
                = r_{t+1} + \gamma_{t+1} 1_{f_{t+1} = 0} G^\infty_{t+1}
        \,. $$

    ATTN this explanation of $\gamma_t$ makes no sense. Either we compute
    expectations and discount by gamma and never terminate (since gamma
    is the probability of termination conditional on NOT having terminated
    before), or terminate upon end-of-episode and never discount.
    """
    # fast branch for zero-step lookahead, i.e. current value-to-go estimates
    if n_lookahead < 1:
        return val

    # get properly broadcasted and zeroed discount coefficients
    # gam_[t] = gamma * (fin[t] == 0) * gamma, t=0..T-1
    fin_, gam_, rew_ = gamma(rew, gam, val, fin=fin)

    # diffably compute the multi-step ahead bootstrapped returns
    val_ = val.clone()  # XXX we will be diffably overwriting some values
    for _ in range(n_lookahead):
        # [O(T B F)] one-step time-delta target $r_{t+1} + \beta_{t+1} v_{t+1}$
        # where $\gamma_{t+1} = \gamma 1_{\neg f_{t+1}}$.
        #     val_[t] = rew[t] + gam_[t] * val[t+1], t=0..T-1
        # here `gam_[t]` is $\beta_{t+1}$ and `rew` is $r_{t+1}$
        val_[:-1] = torch.addcmul(rew_, gam_, val)[1:]

        # `val[-1]` never changes, cause it's zero-step ahead bootstrapped!
        val = val_.clone()
        # XXX l-step ahead bootstrapped return estimate can be though of as
        #    G^l_t = r_{t+1} + \gamma \tilde{v}_{t+1}
        # where $\tilde{v}_{t+1}$ is the (l-1)-step bootstrapped value esimate.
        # Here `val` is exactly that.

    return val


def pyt_q_values(
    qon: Tensor,
    qtg: Tensor = None,
    *,
    double: bool = True,
) -> Tensor:
    r"""Compute the state-values from the q-factors using Double-DQN method.

    Details
    -------
    In Q-learning the action value function $q_{\theta t}$ minimizes
    the TD-error
    $$
        r_{t+1} + \gamma 1_{\neg f_{t+1}} \hat{v}_{t+1}
            - q_{\theta t}(a_t)
        \,, $$

    where $\hat{v}_{t+1}$ is the value estimate of the next state following
    taking the action $a_t$. The error is minimized w.r.t. Q-network parameters
    $\theta$, with $
        q_{\phi t}(\cdot)
            = q((x_t, a_{t-1}, r_t, f_t, h_t), \cdot; \phi)
    $. The difference between various Q-learning methods lies in the way
    the next state value is computed.

    In classic Q-learning there is no target network and the next state
    optimal value function is bootstrapped using the current Q-network:
    $$
        \hat{v}_t \approx \max_a q_{\theta t}(a)
        \,. $$

    The DQN method, proposed by

        [Minh et al. (2013)](https://arxiv.org/abs/1312.5602),

    uses a secondary Q-network to estimate the value of the next state:
    $$
        \hat{v}_t \approx \max_a q_{\theta^- t}(a)
        \,, $$

    where $\theta^-$ are frozen parameters of the target Q-network. The Double
    DQN algorithm of

        [van Hasselt et al. (2015)](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

    unravels the $\max$ operator as $
        \max_k u_k
            \equiv u_{\arg \max_k u_k}
    $ and replaces the outer $u$ with the Q-values of the target net, while
    the inner $u$ (inside the $\arg \max$) is computed with the Q-values of
    the current Q-network. Specifically, the Double DQN value estimate is
    $$
        \hat{v}_t \approx q_{\theta^- t}(\hat{a}_t)
        \,, $$

    for $
        \hat{a}_t = \arg \max_a q_{\theta t}(a)
    $ being the action taken by the current Q-network $\theta$ at $x_t$.
    """  # noqa: E501
    # use $q_{\theta t}$ instead of $q_{\theta^- t}$ in Q-learning
    if qtg is None:
        qtg, double = qon, False

    # get the value estimate from the q-factor
    # `qon` is $q_{\theta t}$, and `qtg` is $q_{\theta^- t}$
    if double:
        # get $\hat{a}_t = \arg \max_a q_{\theta t}$
        hat = qon.argmax(dim=-1, keepdim=True)

        # get $\hat{v}_t = q_{\theta^- t}(\hat{a}_t)$
        val = qtg.gather(-1, hat).squeeze_(-1)

    else:
        # get $\hat{v}_t = \max_a q_{\theta^- t}(a)$
        val = qtg.max(dim=-1).values

    return val


def pyt_td_target(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
) -> Tensor:
    r"""Compute the TD(0) targets.

    Details
    -------
    see `pyt_ret_gae` about synchronization of `rew`, `fin` and `val`.
    `rew` is on element shorter than `val`! See `gamma()` about `gam`.
    """
    return pyt_multistep(rew, val, gam, fin, n_lookahead=1)


def pyt_q_targets(
    rew: Tensor,
    qon: Tensor,
    qtg: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
    *,
    double: bool = True,
    n_lookahead: int = 1,
) -> Tensor:
    """Compute the multistep lookahead Double-DQN targets."""

    # get the state value estimates from the q-values
    val = pyt_q_values(qon, qtg, double=double)

    # compute one- or multi-step lookahead bootstrapped value targets
    return pyt_multistep(rew, val, gam, fin, n_lookahead=n_lookahead)
