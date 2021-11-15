import torch

from rlplay.engine import BaseActorModule
from rlplay.engine.utils import plyr
from rlplay.utils.common import multinomial as base_multinomial


def multinomial(v):
    """Sample a categoricla r.v. from the unnormalized logits in the last dim.
    """
    return base_multinomial(v.detach().softmax(dim=-1))


def postproc(obs, hx, *, val, hlt, **heads):
    r"""Post-process the output of the network to interface with `rlplay`.

    Returns
    -------
    action: tuple
        The action tuple has two components: the halting flag and the action
        itself. The flag is drawn form a bernouli r.v. with sigmoid-transformed
        probabilty of success. The action is a dict of named actions, sampled
        from a categorical distribution in the last dim and possibly masked.

    hx: Tensor or tuple of Tensors
        The recurrent state is kep intact and passed as-is.

    info: dict of Tensors
        The actor's info dict with differentialbe tensor data including halting
        logits, unmasked raw action data, and the value baseline.

    Details
    -------
    To generate the halting flag $
        H_t \sim \mathcal{B}(\{0, 1\}, \sigma(z))
    $ we use the fact that for $
        U \sim \mathcal{U}[0, 1]
    $ we have
    $$
    \sigma(z)
        = \mathbb{P}\bigl(U \leq \sigma(z) \bigr)
        = \mathbb{P}\bigl(\log \frac{U}{1-U} \leq z\bigr)
        \,. $$  % (this is a logisitc r.v.)
    """
    # mask the forbidden actions according to the chassis's aux info
    masked = heads.copy()  # a shallow copy of the heads container
    masked['micro'] = masked['micro'].masked_fill(
        obs['chassis_mask'],  # XXX ooh another hardcode here
        -float('inf'),
    )

    # generate logistic r.v variates to get the halting flag in the end
    tau = torch.empty_like(hlt).uniform_().logit_()
    # XXX inplace Tensor sampler methods accept `generator=...`
    # gen = torch.Generator(device)
    # gen.manual_seed(seed)
    # it is sad that we cannot `.to` generators between devices and register
    #  them in nn.Module for convenience.

    # return composite actions: halt and the action itself
    # XXX we hardcode discrete action here
    return (
        hlt.ge(tau).squeeze(-1),
        plyr.suply(multinomial, masked),
    ), hx, dict(val=val.squeeze(-1), hlt=hlt.squeeze(-1), **heads)
    # XXX `hlt` and `val` in the dict are expected to be T x B
