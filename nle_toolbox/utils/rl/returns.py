import torch


def trailing_broadcast(what, to):
    """Add extra trailing unitary dims to `what` for broadcasting it to `to`.
    """
    return what.reshape(what.shape + (1,) * max(to.ndim - what.ndim, 0))


def pyt_ret_gae(rew, fin, val, *, gam, lam, rho=None):
    r"""Compute the Generalized Advantage Estimates and the Returns.

    Details
    -------
    The sequential data in `rew` [T x B x ...], `fin` [T x B x ...] and
    `val` [(T + 1) x B x ...] is expected to have the following timing: for
    the transition
    $$
        \omega_t, a_t -->> \omega_{t+1}, x_{t+1}, r^E_{t+1}, d_{t+1}
        \,, $$

    `rew[t]` is $r_{t+1}$, `fin[t]` indicates if $\omega_{t+1}$ is terminal,
    and `val[t]` is the bootsrap estimate $v_t$ of the value-to-go from the
    current state $\omega_t$.
    """

    # broadcast gamma coefficients over the logcal-not of the termination mask
    fin_ = trailing_broadcast(fin, rew)
    gam_ = rew.new_full(fin_.shape, gam).masked_fill_(fin_, 0.)

    # [O(T B F)] delta_t = r_{t+1} + \gamma 1_{T \leq t+1} v_{t+1} - v_t
    # td(n) target is the (n+1)-step lookahead value estimate over the current
    #  trajectory. The td(n)-residual can be computed using td(0)-errors
    # \delta^{n+1}_t = \sum_{j=0}^{n+1} \gamma^j \delta_{t+j}  % just expand!
    # \delta_t = r_{t+1} + \gamma v_{t+1} - v_t
    delta = torch.addcmul(rew, gam_, val[1:]).sub_(val[:-1])
    # XXX `bootstrapping` means using v_{t+h} (the func `v(.)` itself) as an
    # approximation of the present value of rewards-to-go (r_{j+1})_{j\geq t+h}

    # A_t = (1 - \lambda) \sum_{t \leq j} \lambda^{j-t} \delta^{j-t}_j
    #     = \sum_{s \geq 0} \delta_{t+s} (\gamma \lambda)^s
    # G_t = r_{t+1} + \gamma G_{t+1} 1_{\neg d_{t+1}}
    # XXX this loop version has slight overhead which is noticeable only for
    # short sequences and small batches, but otherwise this scales linearly
    # in all dimensions. Using doubly buffered
    #   `.addcmul(rew, gam_, ret[j, 1:], out=ret[1-j, :-1])`
    # for `ret` (and similar for `gae`) would increase the complexity of each
    # iteration by T times! (`pyt_returns` and `pyt_multistep_returns`)
    gae, ret = torch.zeros_like(val), val.clone()
    for j in range(1, len(delta) + 1):
        # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
        # t is -j, t+1 is -j-1 (j=1..T)

        # GAE [O(B F)] A_t = \delta_t + \lambda \gamma A_{t+1} 1_{\neg d_{t+1}}
        # gae[t] = delta[t] + gamma * C * fin[t] * gae[t+1], t=0..T-1
        torch.mul(gae[-j], lam * gam, out=gae[-j-1]).masked_fill_(fin_[-j], 0.)
        gae[-j-1].add_(delta[-j])

        # RET [O(B F)] G_t = r_{t+1} + \gamma G_{t+1} 1_{\neg d_{t+1}}
        # ret[t] = rew[t] + gamma * fin[t] * ret[t+1], t=0..T-1
        torch.mul(ret[-j], gam, out=ret[-j-1]).masked_fill_(fin_[-j], 0.)
        ret[-j-1].add_(rew[-j])

    return ret[:-1], gae[:-1]


def pyt_vtrace(
    rew,
    fin,
    val,
    rho,
    *,
    gam,
    r_bar=float('+inf'),
    c_bar=float('+inf'),
):
    r"""Compute the V-trace state-value estimate.

    Details
    -------
    The notation in this brief description deviates from

        [Espeholt et al. (2018)](http://proceedings.mlr.press/v80/espeholt18a.html)

    Specifically we denote the log-likelihood ratio of target and behavior
    policies by $
        \rho_t = \log \pi(a_t \mid x_t) - \log \mu(a_t \mid x_t)
    $, where $\mu$ is the behavior policy and $\pi$ is the target.

    Recall that the TD(0) residuals are given by
    $$
        \delta_t = r_{t+1} + \gamma v_{t+1} 1_{\neg d_{t+1}} - v_t
        \,, $$

    where $v_t = V(\omega_t)$ -- the state-value function associated with
    the behaviour policy $\mu$, and $\delta_s = 0$ for all $s \geq t$ if
    $d_t = \top$. The $n$-step lookahead v-trace value estimate is
    $$
        \hat{v}^n_t
            = v_t + \sum_{j=t}^{t+n-1} \gamma^{j-t}
                       \delta_j \eta_j \prod_{k=t}^{j-1} c_k
        \,, $$

    where $
        c_j = \min\{ e^\rho_j, \bar{c} \}
    $, $
        \eta_j = \min\{ e^\rho_j, \bar{\rho} \}
    $, and $\hat{v}^n_s = 0$ for all $s \geq t$ if $d_t = \top$. At
    the $n \to \infty$ limit (bounded $\rho_t$, $v_t$, and $r_t$) we
    get a forward recurrence
    $$
        \hat{v}_t
            = v_t + \sum_{j \geq t}
                \gamma^{j-t} \delta_j \eta_j \prod_{k=t}^{j-1} c_k
            = v_t + \delta_t \eta_t + \gamma c_t (\hat{v}_{t+1} - v_{t+1})
        \,. $$
    """

    # add extra trailing unitary dims for broadcasting
    # XXX rho is the current/behavioural likelihood ratio for the taken action
    fin_ = trailing_broadcast(fin, rew)
    rho_ = rho.reshape_as(fin_)

    # [O(T B F)] get the clipped importance-weighted td(0)-residuals
    #     \delta_t = r_{t+1} + \gamma v_{t+1} - v_t
    #     \rho_t = \min\{ \bar{\rho},  \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
    gam_ = rew.new_full(fin_.shape, gam).masked_fill_(fin_, 0.)
    delta = torch.addcmul(rew, gam_, val[1:]).sub_(val[:-1])
    delta.mul_(rho_.exp().clamp_(max=r_bar))  # NB extra copy by `.exp()`

    adv = torch.zeros_like(val)

    # c_t = \min\{ \bar{c}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
    see = rho_.exp().clamp_(max=c_bar)
    for j in range(1, len(delta) + 1):
        # V-trace [O(B F)] \hat{v}_t = \hat{a}_t + v_t
        #         \hat{a}_t = \rho_t \delta_t
        #                   + \gamma c_t 1_{\neg d_{t+1}} \hat{a}_{t+1}
        # adv[t] = rho[t] * delta[t] + gamma * fin[t] * see[t] * adv[t+1]
        adv[-j-1].addcmul_(adv[-j], see[-j], value=gam)
        adv[-j-1].masked_fill_(fin_[-j], 0.).add_(delta[-j])

    return adv + val


def pyt_multistep(rew, fin, val, *, gam, n_lookahead=4):
    r"""Compute the multistep lookahead bootstrapped returns.

    Details
    -------
    see `pyt_ret_gae` about synchronization of `rew`, `fin` and `val`.
    `rew` is on element shorter than `val`!

    This computes the $l$-step lookahead bootstrapped return (value estimate):
    $$
        G^l_t
            = \sum_{j=0}^{l-1}
                \biggl\{ \prod_{k < j} \beta_{t+k+1} \biggr\}
                r_{t+k+1}
            + \biggl\{ \prod_{k < l} \beta_{t+k+1} \biggr\}
                v_{t+l}
                % bootstrapped value-to-go estimate at state t+l. It does
                % not include r_{t+l}, since this reward has been received
                % SIMULTANEOUSLY with transitioning to the state.
        \,, $$
    where $
        \beta_t = \gamma 1_{f_t \neq \top}
    $ -- the observed conditional probability of not terminating on
    the t-th step.

    ATTN this explanation of $\beta_t$ makes no sense. Either we compute
    expectations and discount by gamma and never terminate (since gamma
    is the probability of termination conditional on NOT havin terminated
    before), or terminate upon end-of-episode and never discount.
    """

    # v_t ~ G_t = r_{t+1} + \gamma 1_{\neg f_{t+1}} G_{t+1}
    # f_{t+1} indicates if $t+1$ is terminal
    fin_ = trailing_broadcast(fin, rew)

    # broadcast gamma coefficients over the negated termination mask
    # gam_[t] = (~fin[t]) * gamma = 1_{\neg f_{t+1}} \gamma, t=0..T-1
    gam_ = rew.new_full(fin_.shape, gam).masked_fill_(fin_, 0.)

    # fast branch for one-step lookahead, i.e. the TD(0) targets.
    if n_lookahead == 1:
        return torch.addcmul(rew, gam_, val[1:])

    assert n_lookahead > 1

    # diffably compute the multi-step ahead bootstrapped returns
    val_ = val.clone()  # XXX we will be diffably overwriting some values
    for _ in range(1, n_lookahead):
        # [O(T B F)] one-step time-delta target $r_{t+1} + \beta_{t+1} v_{t+1}$
        # where $\beta_{t+1} = \gamma 1_{\neg f_{t+1}}$.
        #     val_[t] = rew[t] + gam_[t] * val[t+1], t=0..T-1
        # here `gam_[t]` is $\beta_{t+1}$ and `rew` is $r_{t+1}$
        val_[:-1] = torch.addcmul(rew, gam_, val[1:])

        # `val[-1]` never changes, cause it's zero-step ahead bootstrapped!
        val = val_.clone()

    # do not cut off incomplete returns
    return torch.addcmul(rew, gam_, val[1:])


def pyt_td_target(rew, fin, val, *, gam):
    r"""Compute the TD(0) targets.

    Details
    -------
    see `pyt_ret_gae` about synchronization of `rew`, `fin` and `val`.
    `rew` is on element shorter than `val`!
    """
    return pyt_multistep(rew, fin, val, gam=gam, n_lookahead=1)


def pyt_q_targets(
    rew,
    fin,
    qon,
    qtg=None,
    *,
    gam,
    double=True,
    n_lookahead=1,
):
    r"""Compute the Double-DQN targets.

    Details
    -------
    In Q-learning the action value function minimizes the TD-error
    $$
        r_{t+1} + \gamma \hat{v}_{t+1} 1_{\neg f_{t+1}}
            - q_{\theta t}(a_t)
        \,, $$

    w.r.t. Q-network parameters $\theta$, with $
        q_{\phi t}(\cdot)
            = q((x_t, a_{t-1}, r_t, f_t, h_t), \cdot; \phi)
    $. In classic Q-learning there is no target network and the next state
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
    """
    # use $q_{\theta t}$ instead of $q_{\theta^- t}$ in Q-learning
    if qtg is None:
        qtg, double = qon, False

    # get the value estimate from the q-factor
    # `qon` is $q_{\theta t}$, and `qtg` is $q_{\theta^- t}$
    if double:
        # get $\hat{a}_t = \arg \max_a q_{\theta t}$
        hat = qon.max(dim=-1).indices.unsqueeze(-1)

        # get $\hat{v}_t = q_{\theta^- t}(\hat{a}_t)$
        val = qtg.gather(-1, hat).squeeze(-1)

    else:
        # get $\hat{v}_t = \max_a q_{\theta^- t}(a)$
        val = qtg.max(dim=-1).values

    # compute one- or multi-step lookahead bootstrapped value targets
    return pyt_multistep(rew, fin, val, gam=gam, n_lookahead=n_lookahead)
