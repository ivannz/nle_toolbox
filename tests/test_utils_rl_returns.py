import pytest

import torch
from typing import Union
from torch import Tensor

from nle_toolbox.utils.rl.returns import trailing_broadcast
from nle_toolbox.utils.rl.returns import pyt_ret_gae, pyt_multistep


def random_data(
    n_seq: int = 256,
    n_env: int = 256,
    *shape,
    binary: bool = False,
    dtype: torch.dtype = torch.float64,
    init_nan: bool = True,
) -> tuple[Tensor, Tensor, Union[float, Tensor], Tensor]:
    # produce some random discount factors, rewards and value estimates
    rew = torch.rand(n_seq, n_env, *shape, dtype=dtype).log_().neg_()
    val = torch.randn(n_seq, n_env, *shape, dtype=dtype)
    gam = torch.rand(n_seq, n_env, *shape, dtype=dtype)

    if init_nan:
        rew[:1] = float("nan")

    # generate ternary fin mask
    fin = torch.randint(-1, 2, size=(n_seq, n_env), dtype=torch.int8)
    return rew, val, gam, (fin.ne(0) if binary else fin)


def broadcast(
    rew: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
) -> tuple[Tensor, Tensor]:
    fin_ = trailing_broadcast(fin, rew)
    if isinstance(gam, float):
        gam_ = rew.new_full(fin_.shape, gam)

    else:
        gam_ = trailing_broadcast(gam, rew).clone()

    return fin_, gam_


def manual_present_value(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
) -> Tensor:
    fin, gam = broadcast(rew, gam, fin)

    # backward recursion from t=T..0
    # R_t = V_T      % if t = T
    #     = r_{t+1}  % if t < T
    #     + \gamma_{t+1} (v_{t+1} 1_{f_{t+1} = -1} + R_{t+1} 1_{f_{t+1} = 0})
    res, ret = [val[-1]], val[-1]
    for t in range(1, len(val)):
        fin_, rew_, val_, gam_ = fin[-t], rew[-t], val[-t], gam[-t]
        ret = rew_ + gam_ * (ret * fin_.eq(0) + val_ * fin_.eq(-1))
        res.append(ret)

    return torch.stack(res[::-1], axis=0)


def manual_deltas(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
) -> Tensor:
    fin, gam = broadcast(rew, gam, fin)

    # \delta_t = r_{t+1}
    #          + \gamma_{t+1} v_{t+1} (1_{f_{t+1} = -1} + 1_{f_{t+1} = 0}) - v_t
    blk = torch.logical_or(fin.eq(0), fin.eq(-1))
    return rew[1:] + gam[1:] * val[1:] * blk[1:] - val[:-1]


def manual_ret_gae(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    lam: Tensor,
    fin: Tensor,
) -> tuple[Tensor, Tensor]:
    ret_ = manual_present_value(rew, val, gam, fin)

    gae_ = manual_present_value(
        manual_deltas(rew, val, gam, fin),
        torch.zeros_like(val),
        gam * lam,
        torch.where(fin.ne(0), 1, 0),
    )
    return ret_, gae_[:-1]


def manual_multistep(
    rew: Tensor,
    val: Tensor,
    gam: Union[float, Tensor],
    fin: Tensor,
    *,
    n_lookahead: int,
) -> Tensor:
    fin, gam = broadcast(rew, gam, fin)

    # compute h-step lookahead value-to-go estimates
    # x^h_t = \sum_{j=0}{^{h-1} \Gamma^j_{t+1} r_{t+j+1} + \Gamma^h_{t+1} v_{t+h}
    # recursion on the horizon h
    # x^0_t = v_t (t = 0..T-1) and x^h_T = v_T (h = 0..H)
    # x^h_t = r_{t+1}  (t = 0..T-1)
    #       + \gamma_{t+1} v_{t+1} 1_{f_{t+1} = -1}
    #       + \gamma_{t+1} x^{h-1}_{t+1} 1_{f_{t+1} = 0}
    rew_ = rew + gam * fin.eq(-1) * val
    ret = val.new_zeros((1 + n_lookahead,) + val.shape).copy_(val)
    for j in range(n_lookahead):
        ret[j + 1, :-1] = rew_[1:] + gam[1:] * fin[1:].eq(0) * ret[j, 1:]

        assert torch.allclose(ret[j + 1, -1:], val[-1:])

    return ret[n_lookahead]


def test_torch_ternary_to_binary(
    shape: tuple[int] = (256, 256),
) -> None:
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
def test_manual_multistep(
    binary: bool,
    n_seq: int = 256,
    n_env: int = 256,
    shape: tuple[int] = (),
) -> None:
    """Sanity check on the tester function."""
    rew, val, gam, fin = random_data(n_seq, n_env, *shape, binary=binary)

    res = manual_multistep(rew, val, gam, fin, n_lookahead=1)
    assert torch.allclose(
        res[:-1],
        (rew + gam * (fin.eq(0) * val + fin.eq(-1) * val))[1:],
    )
    assert torch.allclose(res[-1:], val[-1:])

    res = manual_multistep(rew, val, gam, fin, n_lookahead=3)

    # T-1
    assert torch.allclose(res[-1], val[-1])

    # T-2
    assert torch.allclose(
        res[-2],
        rew[-1] + gam[-1] * (fin[-1].eq(0) * val[-1] + fin[-1].eq(-1) * val[-1]),
    )

    # T-3
    assert torch.allclose(
        res[-3],
        rew[-2]
        + gam[-2]
        * (
            fin[-2].eq(-1) * val[-2]
            + fin[-2].eq(0)
            * (rew[-1] + gam[-1] * (fin[-1].eq(0) * val[-1] + fin[-1].eq(-1) * val[-1]))
        ),
    )

    # 0..T-4
    for t in range(n_seq - 3):
        assert torch.allclose(
            res[t],
            rew[t + 1]
            + gam[t + 1]
            * (
                fin[t + 1].eq(-1) * val[t + 1]
                + fin[t + 1].eq(0)
                * (
                    rew[t + 2]
                    + gam[t + 2]
                    * (
                        fin[t + 2].eq(-1) * val[t + 2]
                        + fin[t + 2].eq(0)
                        * (
                            rew[t + 3]
                            + gam[t + 3]
                            * (
                                fin[t + 3].eq(0) * val[t + 3]
                                + fin[t + 3].eq(-1) * val[t + 3]
                            )
                        )
                    )
                )
            ),
        )


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("n_seq", [1, 1024])
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9, 0.98, 1.0])
def test_pyt_ret_gae(
    binary: bool,
    n_seq: int,
    lam: float,
    n_env: int = 256,
    shape: tuple[int] = (),
) -> None:
    rew, val, gam, fin = random_data(n_seq, n_env, *shape, binary=binary)

    ret, gae = pyt_ret_gae(rew, val, gam, lam, fin)
    ret_, gae_ = manual_ret_gae(rew, val, gam, lam, fin)
    assert torch.allclose(ret, ret_) and torch.allclose(gae, gae_)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("n_seq", [1, 1024])
@pytest.mark.parametrize("n_lookahead", [0, 1, 3, 10])
def test_pyt_multistep(
    binary: bool,
    n_seq: int,
    n_lookahead: int,
    n_env: int = 256,
    shape: tuple[int] = (),
) -> None:
    rew, val, gam, fin = random_data(n_seq, n_env, *shape, binary=binary)

    ret = pyt_multistep(rew, val, gam, fin, n_lookahead=n_lookahead)
    ret_ = manual_multistep(rew, val, gam, fin, n_lookahead=n_lookahead)
    assert torch.allclose(ret, ret_)
