import pytest

import torch
from nle_toolbox.utils.rl.returns import trailing_broadcast, pyt_ret_gae


def random_data(n_seq: int, n_env: int, *shape, binary: bool = False):
    # produce some random discount factors, rewards and value estimates
    rew = torch.rand(n_seq, n_env, *shape).log_().neg_()
    val = torch.randn(n_seq + 1, n_env, *shape)
    gam = torch.rand(n_seq, n_env, *shape)

    # generate ternary fin mask
    fin = torch.randint(-1, 2, size=(n_seq, n_env), dtype=torch.int8)
    return rew, val, gam, (fin.ne(0) if binary else fin)


def broadcast(rew, gam, fin):
    fin_ = trailing_broadcast(fin, rew)
    if isinstance(gam, float):
        gam_ = rew.new_full(fin_.shape, gam)

    else:
        gam_ = trailing_broadcast(gam, rew).clone()

    return fin_, gam_


def manual_present_value(rew, val, gam, fin):
    fin, gam = broadcast(rew, gam, fin)

    # backward recursion from t=T..0
    # R_t = V_T      % if t = T
    #     = r_{t+1}  % if t < T
    #     + \gamma_{t+1} (v_{t+1} 1_{f_{t+1} = -1} + R_{t+1} 1_{f_{t+1} = 0})
    res, ret = [], val[-1]
    for t in range(1, 1 + len(rew)):
        fin_, rew_, val_, gam_ = fin[-t], rew[-t], val[-t], gam[-t]
        ret = rew_ + gam_ * (ret * fin_.eq(0) + val_ * fin_.eq(-1))
        res.append(ret)

    return torch.stack(res[::-1], axis=0)


def manual_deltas(rew, val, gam, fin):
    fin, gam = broadcast(rew, gam, fin)

    # \delta_t = r_{t+1}
    #          + \gamma_{t+1} v_{t+1} (1_{f_{t+1} = -1} + 1_{f_{t+1} = 0}) - v_t
    blk = torch.logical_or(fin.eq(0), fin.eq(-1))
    return rew + gam * val[1:] * blk - val[:-1]


def manual_ret_gae(rew, val, gam, lam, fin):
    ret_ = manual_present_value(rew, val, gam, fin)

    gae_ = manual_present_value(
        manual_deltas(rew, val, gam, fin),
        torch.zeros_like(val),
        gam * lam,
        torch.where(fin.ne(0), 1, 0),
    )
    return ret_, gae_


def test_torch_ternary_to_binary(shape=(256, 256)):
    # get binary from ternary data
    fin3 = torch.randint(-1, 2, size=shape, dtype=torch.int8)
    fin2 = fin3.ne(0)  # fin3 is {-1, 0, +1}, fin2 is {0, 1}

    # converting to binary with a logical function yields {0, 1} ints
    assert torch.logical_or(fin2.eq(0), fin2.eq(1)).all()
    assert fin2.ne(-1).all() and fin2.ge(0).all()

    # False == 0, True == 1, False < True
    assert (fin2.gt(0) == fin2).all()
    assert (fin2.lt(1) == fin2.logical_not()).all()

    assert (fin2.ne(0) == fin2).all()
    assert (fin2.ne(1) == fin2.logical_not()).all()


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("n_seq", [1, 1024])
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9, 0.98, 1.0])
def test_pyt_ret_gae(binary, n_seq, lam, n_env=256, shape=()):
    rew, val, gam, fin = random_data(n_seq, n_env, *shape, binary=binary)

    ret, gae = pyt_ret_gae(rew, val, gam, lam, fin)
    ret_, gae_ = manual_ret_gae(rew, val, gam, lam, fin)
    assert torch.allclose(ret, ret_) and torch.allclose(gae, gae_)
