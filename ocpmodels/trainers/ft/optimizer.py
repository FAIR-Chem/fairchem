import fnmatch
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Literal, Tuple, Union, cast

import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Annotated, TypeVar

from ocpmodels.common.typed_config import Field, TypedConfig
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)

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


class ReduceLROnPlateauConfig(TypedConfig):
    name: Literal["ReduceLROnPlateau"]

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

    warmup: Duration
    decay: Duration

    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = True

    rlp: ReduceLROnPlateauConfig | None = None

    def _to_settings(self, context: OptimizerTrainerContext):
        return LR.LinearWarmupCosineDecaySettings(
            warmup_steps=_duration_to_steps(self.warmup, context),
            total_steps=_duration_to_steps(self.decay, context),
            warmup_factor=self.warmup_start_lr_factor,
            min_lr_factor=self.min_lr_factor,
        )


LRSchedulerConfig = Annotated[
    Union[ReduceLROnPlateauConfig, WarmupCosineDecayRLPSchedulerConfig],
    Field(discriminator="name"),
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


class MultiGroupPerGroupOptimConfig(TypedConfig):
    parameter_patterns: list[str]
    optimizer: AdamWConfig
    lr_scheduler: WarmupCosineDecayRLPSchedulerConfig


class MultiGroupOptimConfig(BaseOptimConfig):
    optim_mode: Literal["multi_group"]

    optimizer: AdamWConfig
    lr_scheduler: WarmupCosineDecayRLPSchedulerConfig
    groups: list[MultiGroupPerGroupOptimConfig]


OptimConfig = Annotated[
    Union[SingleGroupOptimConfig, MultiGroupOptimConfig],
    Field(discriminator="optim_mode"),
]


def _construct_single_group(
    model: nn.Module,
    config: SingleGroupOptimConfig,
    context: OptimizerTrainerContext,
):
    match config.optimizer:
        case AdamWConfig():
            optimizer = optim.AdamW(
                model.parameters(), **config.optimizer._to_ctor_kwargs()
            )
        case _:
            raise NotImplementedError(
                f"Optimizer {config.optimizer} not implemented."
            )

    match config.lr_scheduler:
        case WarmupCosineDecayRLPSchedulerConfig(
            rlp=ReduceLROnPlateauConfig() as rlp_config
        ) as cos_config:
            lr_scheduler = LR.LinearWarmupCosineDecayRLPScheduler(
                optimizer,
                cos_config._to_settings(context),
                rlp_config._to_settings(),
            )
        case WarmupCosineDecayRLPSchedulerConfig(rlp=None) as cos_config:
            lr_scheduler = LR.LinearWarmupCosineDecayScheduler(
                optimizer, cos_config._to_settings(context)
            )
        case ReduceLROnPlateauConfig() as rlp_config:
            lr_scheduler = ReduceLROnPlateau(
                optimizer, **rlp_config._to_ctor_kwargs()
            )
        case _:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented."
            )

    return optimizer, lr_scheduler


def _named_parameters_matching_patterns(
    model: nn.Module,
    patterns: list[str],
    check_requires_grad: bool = True,
):
    for name, param in model.named_parameters():
        if check_requires_grad and not param.requires_grad:
            continue

        if (
            matching_pattern := next(
                (
                    pattern
                    for pattern in patterns
                    if fnmatch.fnmatch(name, pattern)
                ),
                None,
            )
        ) is None:
            continue

        yield name, param, matching_pattern


def _split_parameters(model: nn.Module, pattern_lists: list[list[str]]):
    all_parameters: list[nn.Parameter] = list(model.parameters())

    parameters: list[list[nn.Parameter]] = []
    for patterns in pattern_lists:
        matching = [
            p
            for _, p, _ in _named_parameters_matching_patterns(model, patterns)
        ]
        parameters.append(matching)
        # remove matching parameters from all_parameters
        all_parameters = [
            p for p in all_parameters if all(p is not m for m in matching)
        ]

    return parameters, all_parameters


def _construct_multi_group(
    model: nn.Module,
    config: MultiGroupOptimConfig,
    context: OptimizerTrainerContext,
):
    # Split parameters into groups
    param_groups, remaining_parameters = _split_parameters(
        model, [group.parameter_patterns for group in config.groups]
    )
    # There should be no remaining parameters. If there are, then we throw
    # an error and tell the user to add a fallthrough group "*" to the
    # parameter patterns.
    if remaining_parameters:
        raise ValueError(
            f"{len(remaining_parameters)} parameters were not matched by any group. "
            "Please add a fallthrough group '*' to the parameter patterns."
        )

    match config.optimizer:
        case AdamWConfig():
            optimizer = optim.AdamW(
                [
                    {
                        "params": params,
                        **group_config.optimizer._to_ctor_kwargs(),
                    }
                    for params, group_config in zip(
                        param_groups, config.groups
                    )
                ],
                **config.optimizer._to_ctor_kwargs(),
            )
        case _:
            raise NotImplementedError(
                f"Optimizer {config.optimizer} not implemented."
            )

    # Make sure none of the groups have a ReduceLROnPlateau scheduler
    if any(
        isinstance(group_config.lr_scheduler, ReduceLROnPlateauConfig)
        or (group_config.lr_scheduler.rlp is not None)
        for group_config in config.groups
    ):
        raise ValueError(
            "Per-param group ReduceLROnPlateau scheduler not supported for multi group."
        )

    match config.lr_scheduler:
        case WarmupCosineDecayRLPSchedulerConfig(
            rlp=ReduceLROnPlateauConfig() as rlp_config
        ):
            lr_scheduler = LR.PerParamGroupLinearWarmupCosineDecayRLPScheduler(
                optimizer,
                [
                    group_config.lr_scheduler._to_settings(context)
                    for group_config in config.groups
                ],
                rlp_config._to_settings(),
            )
        case WarmupCosineDecayRLPSchedulerConfig(rlp=None):
            lr_scheduler = LR.PerParamGroupLinearWarmupCosineDecayScheduler(
                optimizer,
                [
                    group_config.lr_scheduler._to_settings(context)
                    for group_config in config.groups
                ],
            )
        case _:
            raise NotImplementedError(
                f"LR scheduler {config.lr_scheduler} not implemented for multi group."
            )

    return optimizer, lr_scheduler


TScheduler = TypeVar("TScheduler", infer_variance=True)


class _LrSchedulerWrapper(Generic[TScheduler]):
    def __init__(self, scheduler: TScheduler):
        super().__init__()

        self.scheduler = scheduler

    def step(self, metrics=None, epoch=None) -> None:
        match self.scheduler:
            case LR.LinearWarmupCosineDecayRLPScheduler() | LR.PerParamGroupLinearWarmupCosineDecayRLPScheduler():
                self.scheduler.rlp_step(metrics, epoch)
            case ReduceLROnPlateau():
                if metrics is None:
                    raise Exception(
                        "Validation set required for ReduceLROnPlateau."
                    )
                self.scheduler.step(metrics, epoch)
            case LR._LRScheduler():
                self.scheduler.step(epoch=epoch)
            case _:
                raise ValueError(f"Invalid scheduler: {type(self.scheduler)}")

    def get_lr(self):
        match self.scheduler:
            case ReduceLROnPlateau():
                return self.scheduler.optimizer.param_groups[0]["lr"]
            case LR._LRScheduler():
                lr = self.scheduler.get_lr()
                if isinstance(lr, list):
                    lr = lr[0]
                return lr
            case _:
                raise ValueError(f"Invalid scheduler: {type(self.scheduler)}")


def load_optimizer(
    model: nn.Module,
    config: OptimConfig,
    context: OptimizerTrainerContext,
):
    match config:
        case SingleGroupOptimConfig():
            optimizer, lr_scheduler = _construct_single_group(
                model, config, context
            )
        case MultiGroupOptimConfig():
            optimizer, lr_scheduler = _construct_multi_group(
                model, config, context
            )
        case _:
            raise ValueError(f"Invalid optimizer config: {config}")

    # Construct EMA
    ema = None
    if config.exponential_moving_average:
        ema = ExponentialMovingAverage(
            model.parameters(),
            config.exponential_moving_average.decay,
        )

    # Wrap lr_scheduler in an object that's compatible with the trainer
    lr_scheduler = _LrSchedulerWrapper(lr_scheduler)

    return optimizer, lr_scheduler, ema
