"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Any
from pathlib import Path

import numpy as np

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_geometric.data import Batch


class LinearReference(nn.Module):
    """Compute a linear reference for target scalar properties"""

    def __init__(
        self, linear_reference: torch.Tensor | None = None, max_num_elements: int = 118
    ):
        """
        Args:
            linear_reference (Tensor): tensor with linear reference values
            max_num_elements (int): max number of elements - 118 is a stretch
        """
        super().__init__()
        self.register_buffer(
            name="linref",
            tensor=linear_reference
            if linear_reference is not None
            else torch.zeros(max_num_elements),
        )

    def get_composition_matrix(self, batch: Batch) -> torch.Tensor:
        """Returns a composition matrix with the number of each element in its atomic number

        Args:
            batch (Batch): a batch of data object with atomic graphs

        Returns:
            torch.Tensor
        """
        data_list = batch.to_data_list()
        composition_matrix = torch.zeros(
            len(data_list), len(self.linref), dtype=torch.int
        )
        for i, data in enumerate(data_list):
            composition_matrix[i] = torch.bincount(
                data.atomic_numbers.int(), minlength=len(self.linref)
            )

        return composition_matrix

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, batch: Batch) -> torch.Tensor:
        offset = torch.zeros(len(batch), dtype=self.lin_ref.dtype).index_add(
            0,
            batch.batch,
            self.lin_ref[batch.atomic_numbers.int()],
        )
        return offset


class Normalizer(nn.Module):
    """Normalize/denormalize a tensor and optionally add a atom reference offset."""

    def __init__(
        self,
        mean: float | torch.Tensor = 0.0,
        std: float | torch.Tensor = 1.0,
    ):
        """tensor is taken as a sample to calculate the mean and std"""
        super().__init__()

        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(std, float):
            std = torch.tensor(std)

        self.register_buffer(name="mean", tensor=mean)
        self.register_buffer(name="std", tensor=std)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def forward(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return self.denorm(normed_tensor)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # check if state dict is legacy state dicts
        if isinstance(state_dict["mean"], float):
            state_dict.update({k: torch.tensor(state_dict[k]) for k in ("mean", "std")})

        super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_normalizer(
    file: str | Path | None = None,
    state_dict: dict | None = None,
    tensor: torch.Tensor | None = None,
    mean: float | torch.Tensor | None = None,
    std: float | torch.Tensor | None = None,
    device: str = "cpu",
) -> Normalizer:
    """Build a target data normalizers with optional atom ref

    Args:
        file (str or Path): path to pt or npz file.
        state_dict (dict): a state dict for Normalizer module
        tensor (Tensor): a tensor with target values used to compute mean and std
        mean (float | Tensor): mean of target data
        std (float | Tensor): std of target data
        device (str): device

    Returns:
        Normalizer
    """
    # path takes priority if given
    if file is not None:
        try:
            # try to load a Normalizer pt file
            state_dict = torch.load(file)
        except RuntimeError:  # try to read an npz file
            # try to load an NPZ file
            values = np.load(file)
            mean = values.get("mean")
            std = values.get("std")

    if state_dict is not None:
        normalizer = Normalizer().load_state_dict(state_dict)
        normalizer.to(device)
        return normalizer

    # if not then read targent value tensor
    if tensor is not None and mean is not None and std is not None:
        mean = torch.mean(tensor, dim=0)
        std = torch.std(tensor, dim=0)
    elif mean is not None and std is not None:
        mean = torch.tensor(mean)
        std = torch.tensor(std)

    # if mean and std are still None than raise an error
    if mean is None or std is None:
        raise ValueError(
            "Incorrect inputs. One of the following sets of inputs must be given: ",
            "a file path to a .pt or .npz file, or mean and std values, or a tensor of target values",
        )

    normalizer = Normalizer(mean=mean, std=std)
    normalizer.to(device)
    return normalizer
