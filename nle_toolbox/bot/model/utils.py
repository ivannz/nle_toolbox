import torch
import plyr

from rlplay.engine import BaseActorModule

from ...utils.nn import multinomial as base_multinomial


def multinomial(v, dim=-1):
    """Sample a categorical r.v. from the unnormalized logits in the given dim."""

    return base_multinomial(v.detach().softmax(dim=dim), 1, dim).squeeze(dim)


def postproc(hx, *, mask=None, val, hlt, **heads):
    r"""Post-process the output of the network to interface with `rlplay`.

    Parameters
    ----------
    hx: Tensor, or container of Tensors
        The recurrent state.

    mask: optional Tensor, or container of Tensor
        The mask of forbidden actions to apply to the action's prelogits.

    val: Tensor
        The critic's value estimate of the current observation.

    hlt: Tensor
        The logit of the halting flag (see details).

    **heads: dict of Tensor
        The action prelogits.

    Returns
    -------
    action: tuple
        The action tuple has two components: the halting flag and the action
        itself. The flag is drawn form a bernouli r.v. with sigmoid-transformed
        probabilty of success. The action is a dict of named actions, sampled
        from a categorical distribution in the last dim and possibly masked.

    hx: Tensor or tuple of Tensors
        The recurrent state is kept intact and passed as-is.

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
    if mask is not None:
        masked["micro"] = masked["micro"].masked_fill(mask, -float("inf"))
    # XXX masked_fill verifies that the mask data is binary {0, 1} and
    #  doesn't care about the exact integer dtype or bool. Does not work
    #  if the mask if a binary fp data though.

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
        (
            hlt.ge(tau).squeeze(-1),
            plyr.suply(multinomial, masked),
        ),
        hx,
        dict(val=val.squeeze(-1), hlt=hlt.squeeze(-1), **heads),
    )
    # XXX `hlt` and `val` in the dict are expected to be T x B


class NeuralActorModule(BaseActorModule):
    """Interface a module with rlplay."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def reset(self, hx, at):
        reset = getattr(self.module, "reset", super().reset)
        return reset(hx, at)

    def forward(self, obs, mask, *, fin=None, hx=None):
        out, hx = self.module(obs, fin=fin, hx=hx)
        return postproc(hx, mask=mask, **out)

    @torch.no_grad()
    def step(
        self,
        stepno,
        obs,
        act,
        rew,
        fin,
        *,
        hx=None,
        virtual=False,
    ):
        return self(*obs, fin=fin, hx=hx)
