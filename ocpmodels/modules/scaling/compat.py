import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .scale_factor import ScaleFactor

ScaleDict = Union[Dict[str, float], Dict[str, torch.Tensor]]


def _load_scale_dict(scale_file: Optional[Union[str, ScaleDict]]):
    """
    Loads scale factors from either:
    - a JSON file mapping scale factor names to scale values
    - a python dictionary pickled object (loaded using `torch.load`) mapping scale factor names to scale values
    - a dictionary mapping scale factor names to scale values
    """
    if not scale_file:
        return None

    if isinstance(scale_file, dict):
        if not scale_file:
            logging.warning("Empty scale dictionary provided to model.")
        return scale_file

    path = Path(scale_file)
    if not path.exists():
        raise ValueError(f"Scale file {path} does not exist.")

    scale_dict: Optional[ScaleDict] = None
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
        return None

    return scale_dict


def load_scales_compat(
    module: nn.Module, scale_file: Optional[Union[str, ScaleDict]]
) -> None:
    scale_dict = _load_scale_dict(scale_file)
    if not scale_dict:
        return

    scale_factors = {
        module.name or name: (module, name)
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
