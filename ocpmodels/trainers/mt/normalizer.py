from contextlib import contextmanager

import torch
from torch_geometric.data import Batch, Data

from .config import NormalizerTargetConfig


def normalizer_transform(config: dict[str, NormalizerTargetConfig]):
    def transform(data: Data):
        nonlocal config
        for target, target_config in config.items():
            if target not in data:
                raise ValueError(f"Target {target} not found in data.")

            data[target] = (
                torch.tensor(data[target])
                if not isinstance(data[target], torch.Tensor)
                else data[target]
            )
            data[f"{target}_unnormalized"] = data[target].clone()
            data[target] = (
                data[target] - target_config.mean
            ) / target_config.std
            data[f"{target}_norm_mean"] = torch.full_like(
                data[target], target_config.mean
            )
            data[f"{target}_norm_std"] = torch.full_like(
                data[target], target_config.std
            )

        return data

    return transform


@contextmanager
def denormalize_context(
    batch_list: list[Batch],
    additional_tensors: list[dict[str, torch.Tensor]],
    task_level_additional_tensors: list[dict[str, torch.Tensor]],
):
    additional_tensors_list = [d.copy() for d in additional_tensors]
    task_level_additional_tensors_list = [
        d.copy() for d in task_level_additional_tensors
    ]

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

            for d in additional_tensors_list:
                additional_value = d.pop(key, None)
                if additional_value is not None:
                    d[key] = (additional_value * std) + mean
            for d in task_level_additional_tensors_list:
                additional_value = d.pop(key, None)
                if additional_value is not None:
                    std = std.unsqueeze(dim=1)
                    mean = mean.unsqueeze(dim=1)
                    d[key] = (additional_value * std) + mean

    yield batch_list, additional_tensors_list, task_level_additional_tensors_list

    for key in norm_keys:
        for batch in batch_list:
            mean = getattr(batch, f"{key}_norm_mean")
            std = getattr(batch, f"{key}_norm_std")
            value = getattr(batch, key)

            value = (value - mean) / std
            setattr(batch, key, value)
