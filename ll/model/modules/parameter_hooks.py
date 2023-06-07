from logging import getLogger
from typing import Callable, cast

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackRegistrarModuleMixin

log = getLogger(__name__)


class ParameterHookModuleMixin(mixin_base_type(CallbackRegistrarModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.after_backward_hooks: list[
            tuple[list[nn.Parameter], Callable[[nn.Parameter], None]]
        ] = []

        def on_after_backward(_trainer: Trainer, pl_module: LightningModule):
            nonlocal self

            config = cast(BaseConfig, pl_module.hparams)
            if not config.trainer.supports_parameter_hooks:
                return

            log.debug(f"Running after_backward hooks...")
            for parameters, hook in self.after_backward_hooks:
                for parameter in parameters:
                    hook(parameter)
            log.debug(
                f"Done running after_backward hooks. (len={len(self.after_backward_hooks)})"
            )

        self.register_callback(on_after_backward=on_after_backward)

    def register_parameter_hook(
        self, parameters: list[nn.Parameter], hook: Callable[[nn.Parameter], None]
    ):
        self.after_backward_hooks.append((parameters, hook))
        log.debug(
            f"Registered after_backward hook {hook} for {len(parameters)} parameters"
        )
