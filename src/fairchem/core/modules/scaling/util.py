from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .scale_factor import ScaleFactor

if TYPE_CHECKING:
    import torch.nn as nn


def ensure_fitted(module: nn.Module, warn: bool = False) -> None:
    for name, child in module.named_modules():
        if not isinstance(child, ScaleFactor) or child.fitted:
            continue
        msg = (
            f"Scale factor {f'{child.name} ({name})' if child.name is not None else name} is not fitted. "
            "Please make sure that you either (1) load a checkpoint with fitted scale factors, "
            "(2) explicitly load scale factors using the `model.scale_file` attribute, or "
            "(3) fit the scale factors using the `fit.py` script."
        )
        if warn:
            logging.warning(msg)
        else:
            raise ValueError(msg)
