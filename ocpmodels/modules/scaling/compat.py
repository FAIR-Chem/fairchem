import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .scale_factor import ScaleFactor


def load_scales_compat(module: nn.Module, scale_file: Optional[str]):
    if not scale_file:
        return

    path = Path(scale_file)
    if not path.exists():
        return

    scale_dict: Optional[Dict[str, float]] = None
    if path.suffix == ".pt":
        scale_dict = torch.load(path)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            scale_dict = json.load(f)

        if isinstance(scale_dict, dict):
            # old json scale factors have a comment field that has the model name
            scale_dict.pop("comment", None)
    else:
        raise ValueError(f"Unsupported scale file extension: {path.suffix}")

    if not scale_dict:
        return

    scale_factors = {
        module.compat_name or name: (module, name)
        for name, module in module.named_modules()
        if isinstance(module, ScaleFactor)
    }
    logging.debug(
        f"Found the following scale factors: {[(k, name) for k, (_, name) in scale_factors.items()]}"
    )
    for name, scale in scale_dict.items():
        if name not in scale_factors:
            logging.warning(f"Scale factor {name} not found in model")
            continue

        scale_module, module_name = scale_factors[name]
        logging.debug(
            f"Loading scale factor {scale} for ({name} => {module_name})"
        )
        scale_module.set_(scale)
