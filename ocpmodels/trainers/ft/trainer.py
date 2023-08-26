from functools import cached_property

from typing_extensions import override

from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import TypeAdapter
from ocpmodels.trainers.base_trainer import BaseTrainer

from .optimizer import OptimConfig, OptimizerTrainerContext, load_optimizer


@registry.register_trainer("ft")
class FTTrainer(BaseTrainer):
    @cached_property
    def optim_config(self):
        return TypeAdapter(OptimConfig).validate_python(
            self.config["optimizer"]
        )

    @override
    def load_optimizer(self) -> None:
        num_steps_per_epoch = len(self.train_loader)
        self.optimizer, self.lr_scheduler, self.ema = load_optimizer(
            self.model,
            self.optim_config,
            OptimizerTrainerContext(num_steps_per_epoch=num_steps_per_epoch),
        )

    @override
    def load_extras(self) -> None:
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm", False)
