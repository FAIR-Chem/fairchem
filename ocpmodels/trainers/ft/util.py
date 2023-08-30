import fnmatch
from logging import getLogger
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys


log = getLogger(__name__)


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    from ocpmodels.trainers.mt.scaling.scale_factor import ScaleFactor

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
    ignore_keys_patterns: list[str],
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

        if (
            pattern := next(
                (
                    fnmatch.fnmatch(full_key_name, p)
                    for p in ignore_keys_patterns
                ),
                None,
            )
        ) is not None:
            log.info(f"Ignoring missing key {full_key_name} due to {pattern}")

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
            log.warning(error_msg)

    return missing_keys, unexpected_keys


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignore_keys_patterns: list[str] = [],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    updated_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if (
            pattern := any(fnmatch.fnmatch(k, p) for p in ignore_keys_patterns)
        ) is not None:
            log.info(f"Ignoring existing key {k} due to {pattern}")
            continue
        updated_state_dict[k] = v

    incompat_keys = module.load_state_dict(updated_state_dict, strict=False)
    return _report_incompat_keys(
        module,
        incompat_keys,
        ignore_keys_patterns,
        strict=strict,
    )
