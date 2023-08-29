import fnmatch
from logging import getLogger
from typing import Any, Generic

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import TypeVar

from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)

from . import lr_scheduler as LR
from .config import (
    AdamWConfig,
    MultiGroupOptimConfig,
    OptimConfig,
    OptimizerTrainerContext,
    ReduceLROnPlateauBaseConfig,
    ReduceLROnPlateauConfig,
    SingleGroupOptimConfig,
    WarmupCosineDecayRLPSchedulerConfig,
)

log = getLogger(__name__)


def _construct_single_group(
    model: nn.Module,
    config: SingleGroupOptimConfig,
    context: OptimizerTrainerContext,
):
    match config.optimizer:
        case AdamWConfig():
            optimizer = AdamW(
                model.parameters(), **config.optimizer._to_ctor_kwargs()
            )
        case _:
            raise NotImplementedError(
                f"Optimizer {config.optimizer} not implemented."
            )

    match config.lr_scheduler:
        case WarmupCosineDecayRLPSchedulerConfig(
            rlp=ReduceLROnPlateauBaseConfig() as rlp_config
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
    all_parameters: list[tuple[str, nn.Parameter]],
    patterns: list[str],
):
    for name, param in all_parameters:
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


def _split_parameters(
    model: nn.Module,
    pattern_lists: list[list[str]],
    check_requires_grad: bool = True,
):
    all_parameters: list[tuple[str, nn.Parameter]] = [
        (n, p)
        for n, p in model.named_parameters()
        if not check_requires_grad or p.requires_grad
    ]

    parameters: list[list[tuple[str, nn.Parameter]]] = []
    for patterns in pattern_lists:
        matching = [
            (n, p)
            for n, p, _ in _named_parameters_matching_patterns(
                all_parameters, patterns
            )
        ]
        # remove matching parameters from all_parameters
        all_parameters = [
            (n, p)
            for n, p in all_parameters
            if all(p is not m for _, m in matching)
        ]
        parameters.append(matching)

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
        n_params_total = sum(p.numel() for _, p in remaining_parameters)
        remaining_parameter_names = [n for n, _ in remaining_parameters]
        raise ValueError(
            f"{len(remaining_parameters)} nn.Parameter objects ({n_params_total} total parameters)"
            " were not matched by any group. "
            f"These parameters are: {' ' .join(remaining_parameter_names)}. "
            "Please add a fallthrough group '*' to the parameter patterns."
        )

    log.critical(f"Constructed the following parameter groups:")
    for group, params in zip(config.groups, param_groups):
        n_params = sum(p.numel() for _, p in params)
        log.critical(
            f"  {group.parameter_patterns}: {n_params} params with config.optimizer={group.optimizer} and config.lr_scheduler={group.lr_scheduler}"
        )

    match config.optimizer:
        case AdamWConfig():
            optimizer = AdamW(
                [
                    {
                        "params": [p for _, p in params],
                        **group_config.optimizer_config(
                            config.optimizer
                        )._to_ctor_kwargs(),
                        "__parameter_patterns": group_config.parameter_patterns,
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

    match config.lr_scheduler:
        case WarmupCosineDecayRLPSchedulerConfig(
            rlp=ReduceLROnPlateauBaseConfig() as rlp_config
        ):
            lr_scheduler = LR.PerParamGroupLinearWarmupCosineDecayRLPScheduler(
                optimizer,
                [
                    group_config._throw_if_rlp(config.lr_scheduler)
                    .lr_scheduler_config(config.lr_scheduler)
                    ._to_settings(context)
                    for group_config in config.groups
                ],
                rlp_config._to_settings(),
            )
        case WarmupCosineDecayRLPSchedulerConfig(rlp=None):
            lr_scheduler = LR.PerParamGroupLinearWarmupCosineDecayScheduler(
                optimizer,
                [
                    group_config._throw_if_rlp(config.lr_scheduler)
                    .lr_scheduler_config(config.lr_scheduler)
                    ._to_settings(context)
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

    def rlp_step(self, metrics: Any):
        match self.scheduler:
            case LR.LinearWarmupCosineDecayRLPScheduler() | LR.PerParamGroupLinearWarmupCosineDecayRLPScheduler():
                self.scheduler.rlp_step(metrics)
            case ReduceLROnPlateau():
                self.scheduler.step(metrics)
            case LR._LRScheduler():
                pass
            case _:
                raise ValueError(f"Invalid scheduler: {type(self.scheduler)}")

    def step(self) -> None:
        match self.scheduler:
            case LR.LinearWarmupCosineDecayRLPScheduler() | LR.PerParamGroupLinearWarmupCosineDecayRLPScheduler():
                self.scheduler.step()
            case ReduceLROnPlateau():
                pass
            case LR._LRScheduler():
                self.scheduler.step()
            case _:
                raise ValueError(f"Invalid scheduler: {type(self.scheduler)}")

    def get_lr_dict(self):
        match self.scheduler:
            case ReduceLROnPlateau():
                return {
                    f"lr_pg{i}": pg["lr"]
                    for i, pg in enumerate(
                        self.scheduler.optimizer.param_groups
                    )
                }
            case LR._LRScheduler():
                lrs = self.scheduler.get_lr()
                if isinstance(lrs, float):
                    return {"lr": lrs}
                return {f"lr_pg{i}": lr for i, lr in enumerate(lrs)}
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
