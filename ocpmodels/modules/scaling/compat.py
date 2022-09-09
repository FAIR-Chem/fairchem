import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, OrderedDict, Union

import torch
import torch.nn as nn

from .scale_factor import ScaleFactor

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys

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
        return scale_file

    path = Path(scale_file)
    if not path.exists():
        return None

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


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    try:
        scale = model.get_submodule(name)
        if not isinstance(scale, ScaleFactor):
            return None
        return scale
    except AttributeError:
        return None


def _report_incompat_keys(
    model: nn.Module,
    keys: "_IncompatibleKeys",
    strict: bool = False,
):
    # filter out the missing scale factor keys for the new scaling factor module
    missing_keys: List[str] = []
    for full_key_name in keys.missing_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is None:
            missing_keys.append(full_key_name)

    # filter out unexpected scale factor keys that remain from the old scaling modules
    unexpected_keys: List[str] = []
    for full_key_name in keys.unexpected_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is None:
            unexpected_keys.append(full_key_name)

    error_msgs = []
    if len(unexpected_keys) > 0:
        error_msgs.insert(
            0,
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            ),
        )
    if len(missing_keys) > 0:
        error_msgs.insert(
            0,
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            ),
        )

    if len(error_msgs) > 0:
        error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
        )
        if strict:
            raise RuntimeError(error_msg)
        else:
            logging.warning(error_msg)


def _patch_module(module: nn.Module):
    # patch the load_state_dict function to ignore the scale factor keys
    # that are no longer used
    module._old_load_state_dict = module.load_state_dict

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        strict: bool = True,
    ):
        incompat_keys = self._old_load_state_dict(state_dict, strict=False)
        _report_incompat_keys(self, incompat_keys, strict=strict)
        return incompat_keys

    module.load_state_dict = load_state_dict.__get__(module, nn.Module)

    logging.info("Patched load_state_dict to ignore scale factor keys")


def _load_scales(module: nn.Module, scale_file: Optional[str]):
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


def load_scales_compat(
    module: nn.Module, scale_file: Optional[str], patch_module: bool = True
):
    _load_scales(module, scale_file)

    if patch_module:
        _patch_module(module)
