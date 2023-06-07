from functools import wraps
from logging import getLogger
from typing import TYPE_CHECKING

from typing_extensions import TypeVar

from ..exception import SkipBatch
from ..model.config import BaseConfig

if TYPE_CHECKING:
    from ..model.base import LightningModuleBase


log = getLogger(__name__)

THparams = TypeVar("THparams", bound=BaseConfig, infer_variance=True)


def _wrap_fn(
    module: "LightningModuleBase[THparams]",
    fn_name: str,
    is_training_step: bool = False,
):
    old_step = getattr(module, fn_name).__func__

    @wraps(old_step)
    def new_step(
        module: "LightningModuleBase[THparams]", batch, batch_idx, *args, **kwargs
    ):
        try:
            return old_step(module, batch, batch_idx, *args, **kwargs)
        except SkipBatch as e:
            log.info(
                f"[{fn_name}] @ [step={module.global_step}, batch={batch_idx}]: Skipping batch due to SkipBatch exception: {e}"
            )

            if is_training_step:
                return module.skip_batch_training_step(
                    batch, batch_idx, *args, **kwargs
                )

    setattr(module, fn_name, new_step.__get__(module))
    log.info(f"Wrapped {fn_name} for skip_batch_exception")


def wrap_lightning_module(module: "LightningModuleBase[THparams]"):
    log.info(
        "Wrapping training_step/validation_step/test_step/predict_step for skip_batch_exception"
    )

    _wrap_fn(module, "training_step", is_training_step=True)
    _wrap_fn(module, "validation_step", is_training_step=False)
    _wrap_fn(module, "test_step", is_training_step=False)
    _wrap_fn(module, "predict_step", is_training_step=False)

    log.info(
        "Wrapped training_step/validation_step/test_step/predict_step for skip_batch_exception"
    )
