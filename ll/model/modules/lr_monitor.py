from logging import getLogger
from typing import cast

from lightning.pytorch.callbacks import LearningRateMonitor
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


class LRMonitorMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _lr_monitor_callback():
            nonlocal self

            config = cast(BaseConfig, self.hparams).trainer.logging
            if not config.log_lr:
                return None

            if self.logger is None:
                log.warning(
                    "Skipping LR logging because no logger is configured. "
                    "Add a logger to your trainer to log learning rates."
                )
                return None

            logging_interval: str | None = None
            if isinstance(config.log_lr, str):
                logging_interval = config.log_lr
            return LearningRateMonitor(logging_interval=logging_interval)

        self.register_callback(_lr_monitor_callback)
