from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal

import torch
import torch.nn.functional as F
from typing_extensions import Annotated

from ocpmodels.common.typed_config import Field, TypedConfig


class LossFnConfig(TypedConfig):
    target: str
    fn: Literal["mae", "mse", "l1", "l2", "l2mae"]

    coefficient: float | list[Any] = 1.0
    reduction: Literal["sum", "mean", "structure_wise_mean"] = "mean"


@dataclass(frozen=True)
class LossFn:
    config: LossFnConfig
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def apply_coefficient(self, loss: torch.Tensor) -> torch.Tensor:
        coefficient = loss.new_tensor(self.config.coefficient)
        return loss * coefficient


LossFnsConfig = Annotated[List[LossFnConfig], Field()]


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
