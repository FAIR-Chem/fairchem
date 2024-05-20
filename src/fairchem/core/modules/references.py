"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch_geometric.data import Batch


class LinearReference(nn.Module):
    """Fit and compute linear reference for target scalar properties"""

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
