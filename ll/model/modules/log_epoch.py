from typing import Protocol, cast, runtime_checkable

from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin


@runtime_checkable
class _HasEpochProperty(Protocol):
    @property
    def epoch(self) -> float:
        ...


def _log_epoch_callback(module: LightningModule, trainer: Trainer, *, prefix: str):
    if trainer.logger is None:
        return

    config = cast(BaseConfig, module.hparams).trainer.logging
    if not config.log_epoch:
        return

    if not isinstance(module, _HasEpochProperty):
        raise TypeError(f"Expected {prefix} to have an epoch property")

    module.log(f"{prefix}epoch", module.epoch, on_step=True, on_epoch=False)


class LogEpochMixin(mixin_base_type(CallbackModuleMixin)):
    @property
    def epoch(self):
        return self.global_step / self.trainer.num_training_batches

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_callback(
            on_train_batch_start=lambda trainer, module, *args, **kwargs: _log_epoch_callback(
                module, trainer, prefix="train/"
            ),
            on_validation_batch_start=lambda trainer, module, *args, **kwargs: _log_epoch_callback(
                module, trainer, prefix="val/"
            ),
            on_test_batch_start=lambda trainer, module, *args, **kwargs: _log_epoch_callback(
                module, trainer, prefix="test/"
            ),
            on_predict_batch_start=lambda trainer, module, *args, **kwargs: _log_epoch_callback(
                module, trainer, prefix="predict/"
            ),
        )
