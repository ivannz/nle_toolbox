"""
The pseudocode below collects fragments of length T by moving through a single
environment in lockstep with a recurrent agent, which keeps its recurrent state
externally in `hx`.

We implement a simple discrete action A2C with GAE, which runs truncated
backprop through time, mixing the recurrent state `hx` with its learnable
initialization `h0` between fragments.
"""
# useful primitives
def collate(fragment): pass  # return map(torch.stack, zip(*fragment))
def gae_ret(rew, fin, val, *, gam, lam): pass  # GAE(gam, lam) and Returns(gam)
def entropy(*, logits): pass  # - \sum_n e^l_{t n} l_{t n}, t=0..T-1
def mse_loss(val, ret): pass  # (v_t - R_t)^2, t=0..T-1
def no_grad(): pass  # disable autodiff within the `with`-scope


# hyperparams
T, n_total = 120, 2_000_000
gam, lam = 0.999, 0.96  # for GAE
C_cr, C_nt = 0.5, 0.01  # for A2C
eta = 0.05  # hx-h0 mixing weight

# setup the env, agent, the initial recurrent state, and an optimizer
env = ...  # an environment with `.reset` and `.step` interface (gym)
agent = ...  # an agent Module, that uses an RNN, e.g. GRU

h0 = Parameter(...)  # the learnable initial state of the agent's RNN
optim = Optimizer([*agent.parameters(), h0])

# init trajectory collection
obs = env.reset()
act = env.action_space.sample()
rew, fin = 0., True
hx = h0

# the outer loop of the A2C
n_steps = 0
while n_steps < n_total:
    # collect a fragment of length T
    fragment = []
    gx = hx
    for t in range(T):  # t=0..T-1 (relative fragment time)
        # ACT: (x_t, a_{t-1}, r_t, d_t), h_t -->> a_t, \log \pi_t, v_t, h_{t+1}
        if fin:
            hx = h0  # h0 if x_t is terminal else h_t
        (upd_act, pol, val), upd_hx = agent(obs, act, rew, fin, hx)
        # XXX agent should not use `act` and `rew` if fin == True

        # save (x_t, a_{t-1}, r_t, d_t, \log \pi_t, v_t), but not h_t!
        # XXX the past recurrent states `hx` live in the heap until backprop
        fragment.append((obs, act, rew, fin, pol, val))

        # deferred hx and act updates
        act, hx = upd_act, upd_hx

        # ENV: x_t, a_t -->> x_{t+1}, r_{t+1}, d_{t+1} (upd. obs, rew, fin)
        # XXX in our notation values with index `t` are CAUSAL PRECURSORS
        # to values with indices `t+1`, hence a_t is rewarded by r_{t+1}!
        obs, rew, fin, nfo = env.step(act)
        if fin:
            obs = env.reset()  # x0 if x_{t+1} is terminal else x_{t+1}

    n_steps += T

    # the 1-step lookahead just for the bootstrap v_T. Don't care about \pi_T.
    (_, pol, val), _ = agent(obs, act, rew, fin, h0 if fin else hx)
    # DO NOT step through env or update `obs`, `act`, `rew`, `fin`, nor `hx`!

    # save (x_T, a_{T-1}, r_T, d_T, \log \pi_T, v_T)
    fragment.append((obs, act, rew, fin, pol, val))
    # XXX here `obs`, `act`, `rew`, `fin` and `hx` are x_T, a_{T-1}, r_T, d_T,
    # and h_T, respectively, which are used to start the next fragment.
    # The data in `obs`, `act`, `rew`, `fin` must be KEPT INTACT, however, to
    # implement T-BPTT, we DO MEDDLE with the "memory" `hx`!

    # collate (x_t, a_{t-1}, r_t, d_t, \log \pi_t, v_t), t=0..T
    tobs, tact, trew, tfin, tpol, tval = collate(fragment)
    # XXX `tpol`, and `tval` are differentiable tensors, others aren't!

    # GAE and returns from (r_t, d_t, v_t)_{t=0}^T (r_0 and d_0 are ignored,
    # but v_0 is used to td-errors in GAE)
    gae, ret = gae_ret(trew, tfin, tval, gam=gam, lam=lam)
    # XXX R_t = r_{t+1} + \gamma R_{t+1}, R_{T-1} = r_T + \gamma v_T

    # policy grad: A_t \log \pi_t(a_t), t=0..T-1
    # XXX a_t sits in `tact[t+1]`, while \log \pi_t -- in `tpol[t]`
    logpr = tpol[:-1].gather(-1, tact[1:].unsqueeze(-1)).squeeze(-1)
    pg = logpr.mul(gae.detach())

    # entropy: - \sum_a \pi_t(a) \log \pi_t(a), t=0..T-1
    nt = entropy(logits=tpol[:-1])

    # critic: v_t \approx R_t = \sum_{j \geq t} \gamma^{j-t} r_{j+1}, t=0..T-1
    cr = mse_loss(tval[:-1], ret)

    # do the backprop
    loss = (- pg - C_nt * nt + (C_cr / 2) * cr).sum()  # max nt, pg; min cr

    optim.zero_grad()
    loss.backward()
    optim.step()

# >>>>>>>>>
    # update the stale state `hx` by doing a non-diffable pass from `gx`
    with no_grad():
        _, hx = agent(tobs, tact, trew, tfin, gx)

    # truncate BPTT, mix h_{T-1} with `h0` to get an h_0 for the next fragment
    hx = hx.detach() * (1 - eta) + h0 * eta  # XXX can also `.lerp`
    # XXX we let some grad feedback from the next fragment into `h0`, in order
    # to learn a, sort of, "stationary" recurrent init.
# <<<<<<<<<
