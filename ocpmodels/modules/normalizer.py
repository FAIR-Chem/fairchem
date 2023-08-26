"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from contextlib import contextmanager
from typing import Dict, List

import torch
from torch_geometric.data import Batch, Data
from typing_extensions import Annotated

from ocpmodels.common.typed_config import Field, TypeAdapter, TypedConfig


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None) -> None:
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

    def to(self, device) -> None:
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict) -> None:
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)


class NormalizerTargetConfig(TypedConfig):
    mean: float = 0.0
    std: float = 1.0


NormalizerConfig = Annotated[Dict[str, NormalizerTargetConfig], Field()]


def normalizer_transform(config_dict: dict):
    config = TypeAdapter(NormalizerConfig).validate_python(config_dict)

    def transform(data: Data):
        nonlocal config
        for target, target_config in config.items():
            if target not in data:
                raise ValueError(f"Target {target} not found in data.")

            data[target] = (
                data[target] - target_config.mean
            ) / target_config.std
            data[f"{target}_norm_mean"] = target_config.mean
            data[f"{target}_norm_std"] = target_config.std

        return data

    return transform


def denormalize_batch(
    batch_list: List[Batch],
    additional_tensors: Dict[str, torch.Tensor] | None = None,
):
    if additional_tensors is None:
        additional_tensors = {}

    keys: set[str] = set([k for batch in batch_list for k in batch.keys])  # type: ignore

    # find all keys that have a norm_mean and norm_std
    norm_keys: set[str] = {
        key.replace("_norm_mean", "")
        for key in keys
        if key.endswith("_norm_mean")
    } & {
        key.replace("_norm_std", "")
        for key in keys
        if key.endswith("_norm_std")
    }

    for key in norm_keys:
        for batch in batch_list:
            mean = getattr(batch, f"{key}_norm_mean")
            std = getattr(batch, f"{key}_norm_std")
            value = getattr(batch, key)

            value = (value * std) + mean
            setattr(batch, key, value)

            additional_value = additional_tensors.pop(key, None)
            if additional_value is not None:
                additional_tensors[key] = (additional_value * std) + mean

    return batch_list, additional_tensors


@contextmanager
def denormalize_context(
    batch_list: List[Batch],
    additional_tensors: Dict[str, torch.Tensor] | None = None,
):
    batch_list, additional_tensors = denormalize_batch(
        batch_list, additional_tensors
    )
    yield batch_list, additional_tensors
    batch_list, additional_tensors = denormalize_batch(
        batch_list, additional_tensors
    )
