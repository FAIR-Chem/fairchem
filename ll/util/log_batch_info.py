from functools import wraps
from logging import getLogger

from lightning.pytorch import LightningModule

from ..exception import SkipBatch, TrainingError

log = getLogger(__name__)


def _wrap_fn(module: LightningModule, fn_name: str):
    old_step = getattr(module, fn_name).__func__

    @wraps(old_step)
    def new_step(module: LightningModule, batch, batch_idx, *args, **kwargs):
        try:
            return old_step(module, batch, batch_idx, *args, **kwargs)
        except BaseException as e:
            if isinstance(e, SkipBatch):
                # we don't need to handle this case
                raise e

            # we need to re-raise the exception with more information
            raise TrainingError(
                e,
                batch_idx=batch_idx,
                batch=batch,
                epoch=module.current_epoch,
                global_step=module.global_step,
                training_fn=fn_name,
            ) from e

    setattr(module, fn_name, new_step.__get__(module))
    log.info(f"Wrapped {fn_name} for log_batch_info")


def wrap_lightning_module(module: LightningModule):
    log.info(
        "Wrapping training_step/validation_step/test_step/predict_step for log_batch_info"
    )

    _wrap_fn(module, "training_step")
    _wrap_fn(module, "validation_step")
    _wrap_fn(module, "test_step")
    _wrap_fn(module, "predict_step")

    log.info(
        "Wrapped training_step/validation_step/test_step/predict_step for log_batch_info"
    )
