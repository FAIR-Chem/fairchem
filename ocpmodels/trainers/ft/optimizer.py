from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from typing_extensions import Annotated, override

from ocpmodels.common.typed_config import Field, TypedConfig
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)


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

    monitor: str
    patience: int
    factor: float
    mode: Literal["min", "max"] = "min"
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"


class WarmupCosineDecayRLPSchedulerConfig(TypedConfig):
    lr_scheduler: Literal["WarmupCosineDecayRLP"]

    warmup: Duration
    decay: Duration

    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = True

    rlp: ReduceLROnPlateauConfig


LRSchedulerConfig = Annotated[
    WarmupCosineDecayRLPSchedulerConfig,
    Field(discriminator="lr_scheduler"),
]

LLRDConfig = Annotated[dict[str, float], Field()]


class ExponentialMovingAverageConfig(TypedConfig):
    decay: float = 0.9999


class OptimConfig(TypedConfig):
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    exponential_moving_average: ExponentialMovingAverageConfig | None = None
    layerwise_learning_rate_decay: LLRDConfig | None = None

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.layerwise_learning_rate_decay:
            # LLRD only works for AdamW + WarmupCosineDecayRLP
            assert (
                self.optimizer.optimizer == "AdamW"
            ), f"LLRD only works for AdamW, not {self.optimizer.optimizer}"
            assert isinstance(
                self.lr_scheduler.lr_scheduler,
                WarmupCosineDecayRLPSchedulerConfig,
            ), f"LLRD only works for WarmupCosineDecayRLP, not {self.lr_scheduler.lr_scheduler}"


def _construct_optimizer(model: nn.Module, config: OptimConfig):
    match config.optimizer:
        case AdamWConfig():
            return optim.AdamW()
        case _:
            raise NotImplementedError(
                f"Optimizer {config.optimizer} not implemented."
            )


def load_optimizer(model: nn.Module, config: OptimConfig):
    # Construct EMA
    ema = None
    if config.exponential_moving_average:
        ema = ExponentialMovingAverage(
            model.parameters(),
            config.exponential_moving_average.decay,
        )

    return ema
