from logging import getLogger
from typing import cast

from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


def _on_train_start_callback(trainer: Trainer, pl_module: LightningModule):
    config = cast(BaseConfig, pl_module.hparams)
    if config.trainer.reduce_lr_on_plateau_sanity_checks == "disable":
        return

    # if no lr schedulers, return
    if not trainer.lr_scheduler_configs:
        return

    errors: list[str] = []
    disable_message = (
        "Otherwise, set `config.trainer.reduce_lr_on_plateau_sanity_checks='disable'` "
        "to disable this sanity check."
    )

    for lr_scheduler_config in trainer.lr_scheduler_configs:
        if not lr_scheduler_config.reduce_on_plateau:
            continue

        match lr_scheduler_config.interval:
            case "epoch":
                # we need to make sure that the trainer runs val every `frequency` epochs

                # If `trainer.check_val_every_n_epoch` is None, then Lightning
                # will run val every `int(trainer.val_check_interval)` steps.
                # So, first we need to make sure that `trainer.val_check_interval` is not None first.
                if trainer.check_val_every_n_epoch is None:
                    errors.append(
                        "Trainer is not running validation at epoch intervals "
                        "(i.e., `trainer.check_val_every_n_epoch` is None) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        f"Please set `config.trainer.check_val_every_n_epoch={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Second, we make sure that the trainer runs val at least every `frequency` epochs
                if (
                    trainer.check_val_every_n_epoch is not None
                    and lr_scheduler_config.frequency % trainer.check_val_every_n_epoch
                    != 0
                ):
                    errors.append(
                        f"Trainer is not running validation every {lr_scheduler_config.frequency} epochs but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} and frequency={lr_scheduler_config.frequency} is used."
                        f"Please set `config.trainer.check_val_every_n_epoch` to a multiple of {lr_scheduler_config.frequency}. "
                        + disable_message
                    )

            case "step":
                # In this case, we need to make sure that the trainer runs val at step intervals
                # that are multiples of `frequency`.

                # First, we make sure that validation is run at step intervals
                if trainer.check_val_every_n_epoch is not None:
                    errors.append(
                        "Trainer is running validation at epoch intervals "
                        "(i.e., `trainer.check_val_every_n_epoch` is not None) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        "Please set `config.trainer.check_val_every_n_epoch=None` "
                        f"and `config.trainer.val_check_interval={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Second, we make sure `trainer.val_check_interval` is an integer
                if not isinstance(trainer.val_check_interval, int):
                    errors.append(
                        f"Trainer is not running validation at step intervals "
                        f"(i.e., `trainer.val_check_interval` is not an integer) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        "Please set `config.trainer.val_check_interval=None` "
                        f"and `config.trainer.val_check_interval={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Third, we make sure that the trainer runs val at least every `frequency` steps
                if (
                    isinstance(trainer.val_check_interval, int)
                    and trainer.val_check_interval % lr_scheduler_config.frequency != 0
                ):
                    errors.append(
                        f"Trainer is not running validation every {lr_scheduler_config.frequency} steps but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} and frequency={lr_scheduler_config.frequency} is used."
                        "Please set `config.trainer.val_check_interval` "
                        f"to a multiple of {lr_scheduler_config.frequency}. "
                        + disable_message
                    )

            case _:
                pass

    if not errors:
        return

    message = (
        "ReduceLRPlateau sanity checks failed with the following errors:\n"
        + "\n".join(errors)
    )
    match config.trainer.reduce_lr_on_plateau_sanity_checks:
        case "warn":
            log.warning(message)
        case "error":
            raise ValueError(message)
        case _:
            pass


class RLPSanityCheckModuleMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global _on_train_start_callback
        self.register_callback(on_train_start=_on_train_start_callback)

    def determine_reduce_lr_on_plateau_interval_frequency(self):
        if (trainer := self._trainer) is None:
            raise RuntimeError(
                "Could not determine the frequency of ReduceLRPlateau scheduler "
                "because `self.trainer` is None."
            )

        # if trainer.check_val_every_n_epoch is an integer, then we run val at epoch intervals
        if trainer.check_val_every_n_epoch is not None:
            return "epoch", trainer.check_val_every_n_epoch

        # otherwise, we run val at step intervals
        if not isinstance(trainer.val_check_batch, int):
            raise ValueError(
                "Could not determine the frequency of ReduceLRPlateau scheduler "
                f"because {trainer.val_check_batch=} is not an integer."
            )
        return "step", trainer.val_check_batch
