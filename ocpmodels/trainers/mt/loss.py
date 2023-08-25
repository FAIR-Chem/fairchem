from functools import partial
from typing import Callable, Dict, List, Literal, TypedDict

import torch
import torch.nn.functional as F
from typing_extensions import NotRequired

from ocpmodels.common.typed_config import TypedConfig


class LossFn(TypedDict):
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    coefficient: NotRequired[float]
    task_idx: NotRequired[int]


class SingleLossFnConfig(TypedConfig):
    fn: Literal["mae", "mse", "l1", "l2", "l2mae"] = "mae"
    coefficient: float = 1.0


class TaskLossFnConfig(TypedConfig):
    losses: Dict[str, SingleLossFnConfig]


class LossFnsConfig(TypedConfig):
    tasks: List[TaskLossFnConfig]
    reduction: Dict[str, Literal["sum", "mean", "structure_wise_mean"]]


def _create_loss(config: SingleLossFnConfig, task_idx: int) -> LossFn:
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

    loss: LossFn = {
        "fn": loss_fn,
        "coefficient": config.coefficient,
        "task_idx": task_idx,
    }
    return loss


def _create_task_losses(config: TaskLossFnConfig, task_idx: int):
    for target_name, loss_config in config.losses.items():
        yield target_name, _create_loss(loss_config, task_idx)


def create_losses(config: LossFnsConfig):
    for task_idx, task_config in enumerate(config.tasks):
        for target_name, loss in _create_task_losses(task_config, task_idx):
            yield target_name, loss
