from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Annotated, override

from ocpmodels.common.typed_config import Field, TypedConfig
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from .lr_scheduler import warmup_cosine_decay


class AdamWConfig(TypedConfig):
    optimizer: Literal["AdamW"]
    lr: float
    amsgrad: bool = False
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


OptimizerConfig = Annotated[AdamWConfig, Field(discriminator="optimizer")]

Duration = Tuple[int, Literal["steps", "epochs"]]


class ReduceLROnPlateauConfig(TypedConfig):
    lr_scheduler: Literal["ReduceLROnPlateau"]

    patience: int
    factor: float
    mode: Literal["min", "max"] = "min"
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"

    def _to_linear_warmup_cos_rlp_dict(self):
        """
        Params for PerParamGroupLinearWarmupCosineAnnealingRLPLR's RLP
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        """
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": False,
        }


class WarmupCosineDecayRLPSchedulerConfig(TypedConfig):
    lr_scheduler: Literal["WarmupCosineDecayRLP"]

    warmup: Duration
    decay: Duration

    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = True

    rlp: ReduceLROnPlateauConfig | None = None


LRSchedulerConfig = Annotated[
    Union[ReduceLROnPlateauConfig, WarmupCosineDecayRLPSchedulerConfig],
    Field(discriminator="lr_scheduler"),
]

LLRDConfig = Annotated[dict[str, float], Field()]


class ExponentialMovingAverageConfig(TypedConfig):
    decay: float = 0.9999


class BaseOptimConfig(TypedConfig):
    exponential_moving_average: ExponentialMovingAverageConfig | None = None


class SingleGroupOptimConfig(BaseOptimConfig):
    optim_mode: Literal["single_group"] = "single_group"

    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


class MultiGroupOptimConfig(BaseOptimConfig):
    optim_mode: Literal["multi_group"]

    groups: dict[str, SingleGroupOptimConfig]


OptimConfig = Annotated[
    Union[SingleGroupOptimConfig, MultiGroupOptimConfig],
    Field(discriminator="optim_mode"),
]


def _construct_single_group(model: nn.Module, config: SingleGroupOptimConfig):
    match config.optimizer:
        case AdamWConfig():
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.optimizer.lr,
                amsgrad=config.optimizer.amsgrad,
                weight_decay=config.optimizer.weight_decay,
                betas=config.optimizer.betas,
                eps=config.optimizer.eps,
            )
        case _:
            raise NotImplementedError(
                f"Optimizer {config.optimizer} not implemented."
            )

    match config.lr_scheduler:
        case WarmupCosineDecayRLPSchedulerConfig():
            lr_scheduler = WarmupCosineDecayRLPScheduler(
                optimizer,
                config.lr_scheduler.warmup,
                config.lr_scheduler.decay,
                warmup_start_lr_factor=config.lr_scheduler.warmup_start_lr_factor,
                min_lr_factor=config.lr_scheduler.min_lr_factor,
                last_step=config.lr_scheduler.last_step,
                should_restart=config.lr_scheduler.should_restart,
            )
        case ReduceLROnPlateauConfig():
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                patience=config.lr_scheduler.patience,
                factor=config.lr_scheduler.factor,
                mode=config.lr_scheduler.mode,
                min_lr=config.lr_scheduler.min_lr,
                eps=config.lr_scheduler.eps,
                cooldown=config.lr_scheduler.cooldown,
                threshold=config.lr_scheduler.threshold,
                threshold_mode=config.lr_scheduler.threshold_mode,
            )
        case _:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented."
            )

    return optimizer, lr_scheduler


def load_optimizer(model: nn.Module, config: OptimConfig):
    match config:
        case SingleGroupOptimConfig():
            optimizer, lr_scheduler = _construct_single_group(
                model, config.optimizer
            )
        case MultiGroupOptimConfig():
            pass
        case _:
            raise ValueError(f"Invalid optimizer config: {config}")

    # Construct EMA
    ema = None
    if config.exponential_moving_average:
        ema = ExponentialMovingAverage(
            model.parameters(),
            config.exponential_moving_average.decay,
        )

    return ema
