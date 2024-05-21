"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
        self.lin_ref = (
            linear_reference
            if linear_reference is not None
            else torch.zeros(max_num_elements)
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
            len(data_list), len(self.lin_ref), dtype=torch.int
        )
        for i, data in enumerate(data_list):
            composition_matrix[i] = torch.bincount(
                data.atomic_numbers.int(), minlength=len(self.lin_ref)
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
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        atomref: nn.Module | None = None,
        device: str = "cpu",
    ):
        """tensor is taken as a sample to calculate the mean and std"""
        super().__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        self.atomref = atomref
        self.to(device)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def forward(self, normed_tensor: torch.Tensor, batch: Batch) -> torch.Tensor:
        tensor = self.denorm(normed_tensor)
        if self.atomref is not None:
            tensor += self.atomref(batch)
        return tensor

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict) -> None:
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)


def build_normalizer(
    file: str | Path | None = None,
    tensor: torch.Tensor | None = None,
    mean: float | torch.Tensor | None = None,
    std: float | torch.Tensor | None = None,
    atomref: torch.Tensor | None = None,
    device: str = "cpu",
) -> Normalizer:
    """Build a target data normalizers with optional atom ref

    Args:
        file (str or Path): path to pt or npz file.
        tensor (Tensor): a tensor with target values used to compute mean and std
        mean (float | Tensor): mean of target data
        std (float | Tensor): std of target data
        atomref (Tensor): tensor of linear atomic reference values
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
            atomref = values.get("atomref")
        else:
            normalizer = Normalizer.load_state_dict(state_dict)
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

    if atomref is not None:
        atomref = LinearReference(atomref)

    return Normalizer(mean=mean, std=std, atomref=atomref, device=device)
