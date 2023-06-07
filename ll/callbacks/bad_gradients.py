from logging import getLogger
from typing import Callable

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from typing_extensions import override

log = getLogger(__name__)


def print_bad_gradients(module: LightningModule):
    for name, param in module.named_parameters():
        if (param.grad is None) or torch.isfinite(param.grad.float()).all():
            continue

        has_nan = torch.isnan(param.grad.float()).any()
        has_inf = torch.isinf(param.grad.float()).any()
        kinds = [
            "NaN" if has_nan else None,
            "Inf" if has_inf else None,
        ]
        kinds = " and ".join(prop for prop in kinds if prop is not None)
        log.critical(f"{name} ({param.shape}) has {kinds} gradients")


class PrintBadGradientsCallback(Callback):
    def __init__(self, *, enabled: Callable[[], bool] = lambda: True):
        super().__init__()

        self.enabled = enabled

    @override
    def on_after_backward(self, _trainer: Trainer, module: LightningModule):
        if not self.enabled():
            return
        print_bad_gradients(module)
