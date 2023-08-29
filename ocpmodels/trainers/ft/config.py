import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Tuple, Union, cast

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Annotated, override

from ocpmodels.common.typed_config import Field, TypedConfig

from . import lr_scheduler as LR


@dataclass
class OptimizerTrainerContext:
    num_steps_per_epoch: int


class AdamWConfig(TypedConfig):
    name: Literal["AdamW"]
    lr: float
    amsgrad: bool = False
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def _to_ctor_kwargs(self):
        if TYPE_CHECKING:
            ctor = partial(AdamW, cast(Any, None))
        else:
            ctor = dict

        return cast(
            dict[str, Any],
            ctor(
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps,
            ),
        )


OptimizerConfig = Annotated[AdamWConfig, Field(discriminator="name")]

Duration = Tuple[int, Literal["steps", "epochs"]]


def _duration_to_steps(duration: Duration, context: OptimizerTrainerContext):
    match duration:
        case (num, "steps"):
            return num
        case (num, "epochs"):
            return num * context.num_steps_per_epoch
        case _:
            raise ValueError(f"Invalid duration: {duration}")


class ReduceLROnPlateauBaseConfig(TypedConfig):
    patience: int
    factor: float
    mode: Literal["min", "max"] = "min"
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"

    def _to_settings(self):
        return LR.ReduceLROnPlateauSettings(
            patience=self.patience,
            factor=self.factor,
            mode=self.mode,
            min_lr=self.min_lr,
            eps=self.eps,
            cooldown=self.cooldown,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
        )


class ReduceLROnPlateauConfig(ReduceLROnPlateauBaseConfig):
    name: Literal["ReduceLROnPlateau"]

    def _to_ctor_kwargs(self):
        if TYPE_CHECKING:
            ctor = partial(ReduceLROnPlateau, cast(Any, None))
        else:
            ctor = dict

        return cast(
            dict[str, Any],
            ctor(
                patience=self.patience,
                factor=self.factor,
                mode=self.mode,
                min_lr=self.min_lr,
                eps=self.eps,
                cooldown=self.cooldown,
                threshold=self.threshold,
                threshold_mode=self.threshold_mode,
            ),
        )


class WarmupCosineDecayRLPSchedulerConfig(TypedConfig):
    name: Literal["WarmupCosineDecayRLP"]

    warmup_duration: Duration
    warmup_factor: float = 0.0

    decay_duration: Duration
    decay_factor: float = 1.0e-2

    rlp: ReduceLROnPlateauBaseConfig | None = None

    def _to_settings(self, context: OptimizerTrainerContext):
        return LR.LinearWarmupCosineDecaySettings(
            warmup_steps=_duration_to_steps(self.warmup_duration, context),
            warmup_factor=self.warmup_factor,
            decay_steps=_duration_to_steps(self.decay_duration, context),
            min_lr_factor=self.decay_factor,
        )


LRSchedulerConfig = Annotated[
    Union[ReduceLROnPlateauConfig, WarmupCosineDecayRLPSchedulerConfig],
    Field(discriminator="name"),
]

LLRDConfig = Annotated[dict[str, float], Field()]


class ExponentialMovingAverageConfig(TypedConfig):
    decay: float


class BaseOptimConfig(TypedConfig):
    exponential_moving_average: ExponentialMovingAverageConfig | None = None


class SingleGroupOptimConfig(BaseOptimConfig):
    optim_mode: Literal["single_group"] = "single_group"

    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


class MultiGroupPerGroupOptimConfig(TypedConfig):
    parameter_patterns: list[str]
    optimizer: dict[str, Any] | None = None
    lr_scheduler: dict[str, Any] | None = None

    def optimizer_config(self, base: AdamWConfig):
        config = base
        if self.optimizer:
            config = cast(
                AdamWConfig,
                config._as_pydantic_model.model_copy(
                    update=self.optimizer, deep=True
                ),
            )
        return config

    def lr_scheduler_config(self, base: WarmupCosineDecayRLPSchedulerConfig):
        config = copy.deepcopy(base)
        if self.lr_scheduler:
            config = cast(
                WarmupCosineDecayRLPSchedulerConfig,
                config._as_pydantic_model.model_copy(update=self.lr_scheduler),
            )
        return config

    def _throw_if_rlp(self, base: WarmupCosineDecayRLPSchedulerConfig):
        if not self.lr_scheduler:
            return self

        config = copy.deepcopy(base)
        config.rlp = None
        lr_scheduler = self.lr_scheduler_config(base)
        if lr_scheduler.rlp is not None:
            raise ValueError(
                "Per-param group ReduceLROnPlateau scheduler not supported for multi group, "
                f"but got {lr_scheduler.rlp=}."
            )
        return self


class MultiGroupOptimConfig(BaseOptimConfig):
    optim_mode: Literal["multi_group"]

    optimizer: AdamWConfig
    lr_scheduler: WarmupCosineDecayRLPSchedulerConfig
    groups: list[MultiGroupPerGroupOptimConfig]


OptimConfig = Annotated[
    Union[SingleGroupOptimConfig, MultiGroupOptimConfig],
    Field(discriminator="optim_mode"),
]


class FinetuneCheckpointConfig(TypedConfig):
    src: Path
    ignore_keys_patterns: list[str] = []


class FinetuneConfig(TypedConfig):
    base_checkpoint: FinetuneCheckpointConfig

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.base_checkpoint.src.exists():
            raise ValueError(
                f"Base checkpoint {self.base_checkpoint} not found."
            )
