from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

from .config import LossFn, LossFnConfig, LossFnsConfig


def _create_loss(config: LossFnConfig) -> LossFn:
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if config.fn in ("mae", "l1"):
        loss_fn = partial(F.l1_loss, reduction="none")
    elif config.fn in ("mse", "l2"):
        loss_fn = partial(F.mse_loss, reduction="none")
    elif config.fn in ("l2mae",):
        loss_fn = F.pairwise_distance
    else:
        raise NotImplementedError(f"{config.fn=} not implemented.")

    # loss_fn = DDPLoss(loss_fn, config.fn, config.reduction)
    # DDP loss is not implemented for MT yet

    return LossFn(config, loss_fn)


def create_losses(config: LossFnsConfig):
    return [_create_loss(loss_config) for loss_config in config]
