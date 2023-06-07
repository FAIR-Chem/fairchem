from logging import getLogger
from typing import cast

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


class WandbWrapperMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def setup(trainer: Trainer, pl_module: LightningModule, stage: str):
            nonlocal self

            config = cast(BaseConfig, self.hparams)
            if (
                not config.trainer.logging.enabled
                or not config.trainer.logging.wandb
                or not config.trainer.logging.wandb.watch
                or not config.trainer.logging.wandb.watch.enabled
            ):
                return

            if (
                logger := next(
                    (l for l in trainer.loggers if isinstance(l, WandbLogger)), None
                )
            ) is None:
                log.warning("Could not find wandb logger or module to log")
                return

            if (module := self.wandb_log_module()) is None:
                log.warning("Could not find module to log to wandb")
                return

            if getattr(self, "_model_watched", False):
                return

            logger.watch(
                module,
                log=cast(str, config.trainer.logging.wandb.watch.log),
                log_freq=config.trainer.logging.wandb.watch.log_freq,
                log_graph=config.trainer.logging.wandb.watch.log_graph,
            )
            setattr(self, "_model_watched", True)

        self.register_callback(setup=setup)

    def wandb_log_module(self) -> nn.Module | None:
        return self
