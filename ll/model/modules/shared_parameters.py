from logging import getLogger
from typing import cast

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackRegistrarModuleMixin

log = getLogger(__name__)


class SharedParametersModuleMixin(mixin_base_type(CallbackRegistrarModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_parameters: list[tuple[nn.Parameter, int | float]] = []

        def on_after_backward(_trainer: Trainer, pl_module: LightningModule):
            nonlocal self

            config = cast(BaseConfig, pl_module.hparams)
            if not config.trainer.supports_shared_parameters:
                return

            log.debug(f"Scaling {len(self.shared_parameters)} shared parameters...")
            warned_shared_param_no_grad = False
            for p, factor in self.shared_parameters:
                if not hasattr(p, "grad") or p.grad is None:
                    warned_shared_param_no_grad = True
                    continue

                _ = p.grad.data.div_(factor)

            if warned_shared_param_no_grad:
                log.warning(
                    "Some shared parameters do not have a gradient. "
                    "Please check if all shared parameters are used "
                    "and point to PyTorch parameters."
                )

            log.debug(
                f"Done scaling shared parameters. (len={len(self.shared_parameters)})"
            )

        self.register_callback(on_after_backward=on_after_backward)

    def register_shared_parameters(
        self, parameters: list[tuple[nn.Parameter, int | float]]
    ):
        for parameter, factor in parameters:
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("Shared parameters must be PyTorch parameters")
            if not isinstance(factor, (int, float)):
                raise ValueError("Factor must be an integer or float")

            self.shared_parameters.append((parameter, factor))

        log.info(f"Registered {len(parameters)} shared parameters")
