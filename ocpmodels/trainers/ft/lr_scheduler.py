import math
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from typing_extensions import override

log = getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LinearWarmupCosineDecaySettings:
    warmup_steps: int
    warmup_factor: float
    decay_steps: int
    min_lr_factor: float

    @property
    def total_steps(self):
        return self.warmup_steps + self.decay_steps


def linear_warmup_cosine_decay_schedule(
    step: int,
    config: LinearWarmupCosineDecaySettings,
):
    # Linear warmup
    if step < config.warmup_steps:
        return config.warmup_factor + (1.0 - config.warmup_factor) * (
            float(step) / float(max(1, config.warmup_steps - 1))
        )

    # Decay following cosine schedule (no restarts/annealing)
    progress = float(step - config.warmup_steps) / float(
        max(1, config.decay_steps - 1)
    )
    # No restarts/annealing
    progress = min(progress, 1.0)
    return max(
        config.min_lr_factor,
        0.5
        * (1.0 + math.cos(math.pi * progress))
        * (1.0 - config.min_lr_factor)
        + config.min_lr_factor,
    )


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        config: LinearWarmupCosineDecaySettings,
        last_epoch: int = -1,
    ):
        self.config = config
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self):
        return [
            lr
            * linear_warmup_cosine_decay_schedule(self.last_epoch, self.config)
            for lr in self.base_lrs
        ]


class PerParamGroupLinearWarmupCosineDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        param_group_configs: list[LinearWarmupCosineDecaySettings],
        last_epoch: int = -1,
    ):
        assert len(param_group_configs) == len(
            optimizer.param_groups
        ), f"The number of param group configs ({len(param_group_configs)}) must match the number of param groups ({len(optimizer.param_groups)})"

        self.param_group_configs = param_group_configs
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self):
        return [
            lr * linear_warmup_cosine_decay_schedule(self.last_epoch, config)
            for lr, config in zip(self.base_lrs, self.param_group_configs)
        ]


@dataclass(frozen=True, kw_only=True)
class ReduceLROnPlateauSettings:
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8
    verbose: bool = False


class LinearWarmupCosineDecayRLPScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        linear_warmup_cosine_decay_config: LinearWarmupCosineDecaySettings,
        rlp_config: ReduceLROnPlateauSettings,
        last_epoch: int = -1,
    ):
        self.total_steps = linear_warmup_cosine_decay_config.total_steps

        self.reduce_lr_on_plateau = ReduceLROnPlateau(
            optimizer, **asdict(rlp_config)
        )
        self.linear_warmup_cosine_decay = LinearWarmupCosineDecayScheduler(
            optimizer,
            linear_warmup_cosine_decay_config,
            last_epoch,
        )
        super().__init__(optimizer, last_epoch)

    @property
    def current_stage(self):
        return (
            "linear_warmup_cosine_decay"
            if self.last_epoch <= self.total_steps
            else "reduce_lr_on_plateau"
        )

    @override
    def get_lr(self):
        return self.linear_warmup_cosine_decay.get_lr()

    def rlp_step(self, metrics: Any, epoch: int | None = None):
        if self.current_stage != "reduce_lr_on_plateau":
            # The caller should be using `step` instead of `rlp_step`
            # to step the warmup and decay schedulers.
            return
        self.reduce_lr_on_plateau.step(metrics, epoch)

    @override
    def step(self, epoch: int | None = None):
        if self.current_stage != "linear_warmup_cosine_decay":
            # The caller should be using `rlp_step` instead of `step`
            # to step the RLP scheduler.
            return
        self.linear_warmup_cosine_decay.step(epoch)

    @override
    def state_dict(self):
        return {
            "warmup_cosine_decay": self.linear_warmup_cosine_decay.state_dict(),
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau.state_dict(),
        }

    @override
    def load_state_dict(self, state_dict):
        self.linear_warmup_cosine_decay.load_state_dict(
            state_dict["warmup_cosine_decay"]
        )
        self.reduce_lr_on_plateau.load_state_dict(
            state_dict["reduce_lr_on_plateau"]
        )


class PerParamGroupLinearWarmupCosineDecayRLPScheduler(
    LinearWarmupCosineDecayRLPScheduler
):
    def __init__(
        self,
        optimizer: Optimizer,
        param_group_linear_warmup_cosine_decay_configs: list[
            LinearWarmupCosineDecaySettings
        ],
        rlp_config: ReduceLROnPlateauSettings,
        last_epoch: int = -1,
    ):
        self.total_steps = next(
            iter(param_group_linear_warmup_cosine_decay_configs)
        ).total_steps
        # Make sure all param groups have the same total steps
        assert all(
            config.total_steps == self.total_steps
            for config in param_group_linear_warmup_cosine_decay_configs
        ), "All param group configs must have the same total steps"

        self.reduce_lr_on_plateau = ReduceLROnPlateau(
            optimizer, **asdict(rlp_config)
        )
        self.linear_warmup_cosine_decay = (
            PerParamGroupLinearWarmupCosineDecayScheduler(
                optimizer,
                param_group_linear_warmup_cosine_decay_configs,
                last_epoch,
            )
        )

        # Call the super of the grandparent class, ignoring the parent class
        # (LinearWarmupCosineDecayRLPScheduler)
        super(LinearWarmupCosineDecayRLPScheduler, self).__init__(
            optimizer, last_epoch
        )
