import logging
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn

from .scale_factor import ScaleFactor

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys


def ensure_fitted(module: nn.Module, warn: bool = False):
    for name, child in module.named_modules():
        if not isinstance(child, ScaleFactor) or child.fitted:
            continue
        if child.name is not None:
            name = f"{child.name} ({name})"
        msg = (
            f"Scale factor {name} is not fitted. "
            "Please make sure that you either (1) load a checkpoint with fitted scale factors, "
            "(2) explicitly load scale factors using the `model.scale_file` attribute, or "
            "(3) fit the scale factors using the `fit.py` script."
        )
        if warn:
            logging.warning(msg)
        else:
            raise ValueError(msg)


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    from ocpmodels.modules.scaling.scale_factor import ScaleFactor

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
) -> tuple[list[str], list[str]]:
    # filter out the missing scale factor keys for the new scaling factor module
    missing_keys: list[str] = []
    for full_key_name in keys.missing_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is not None:
            continue
        missing_keys.append(full_key_name)

    # filter out unexpected scale factor keys that remain from the old scaling modules
    unexpected_keys: list[str] = []
    for full_key_name in keys.unexpected_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is not None:
            continue
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

    return missing_keys, unexpected_keys


def ensure_state_dict_matches(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> None:
    module_state_dict = module.state_dict()
    shared_keys = set(module_state_dict.keys()) & set(state_dict.keys())
    if strict:
        missing_keys = set(module_state_dict.keys()) - shared_keys
        unexpected_keys = set(state_dict.keys()) - shared_keys
        if len(missing_keys) > 0:
            raise RuntimeError(
                f"Missing keys in state_dict: {missing_keys}. "
                "Please make sure that the model has been initialized properly."
            )

        if len(unexpected_keys) > 0:
            raise RuntimeError(
                f"Unexpected keys in state_dict: {unexpected_keys}. "
                "Please make sure that the model has been initialized properly."
            )

    for key in shared_keys:
        state_dict_value = state_dict[key]
        module_value = module_state_dict[key]
        if state_dict_value.shape != module_value.shape:
            raise RuntimeError(
                f"Shape mismatch for key {key}: "
                f"state_dict: {state_dict_value.shape}, "
                f"module: {module_value.shape}. "
                "Please make sure that the model has been initialized properly."
            )

        if state_dict_value.dtype != module_value.dtype:
            raise RuntimeError(
                f"Dtype mismatch for key {key}: "
                f"state_dict: {state_dict_value.dtype}, "
                f"module: {module_value.dtype}. "
                "Please make sure that the model has been initialized properly."
            )

        if not torch.allclose(state_dict_value, module_value):
            raise RuntimeError(
                f"Value mismatch for key {key}: "
                f"state_dict: {state_dict_value}, "
                f"module: {module_value}. "
                "Please make sure that the model has been initialized properly."
            )


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
    check_matches: bool = True,
) -> tuple[list[str], list[str]]:
    incompat_keys = module.load_state_dict(state_dict, strict=False)  # type: ignore
    missing_keys, unexpected_keys = _report_incompat_keys(
        module, incompat_keys, strict=strict
    )
    if check_matches:
        ensure_state_dict_matches(module, state_dict, strict=strict)
    return missing_keys, unexpected_keys
