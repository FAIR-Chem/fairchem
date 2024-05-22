"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

    from torch_geometric.data import Batch


class LinearReference(nn.Module):
    """Compute a linear reference for target scalar properties"""

    def __init__(
        self,
        element_references: torch.Tensor | None = None,
        max_num_elements: int = 118,
    ):
        """
        Args:
            element_references (Tensor): tensor with linear reference values
            max_num_elements (int): max number of elements - 118 is a stretch
        """
        super().__init__()
        self.register_buffer(
            name="elementref",
            tensor=element_references
            if element_references is not None
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
    def forward(self, target: torch.Tensor, batch: Batch) -> torch.Tensor:
        return target.to(self.elementref.dtype).index_add(
            0,
            batch.batch,
            self.elementref[batch.atomic_numbers.int()],
        )


def create_element_references(
    type: Literal["linear"] = "linear",
    file: str | Path | None = None,
    state_dict: dict | None = None,
    device: str = "cpu",
) -> LinearReference:
    """Create an element reference module.

    Currently only linear references are supported.

    Args:
        file (str or Path): path to pt or npz file
        state_dict (dict): a state dict of a element reference module
        device (str): device to move element ref into

    Returns:
        LinearReference
    """

    # path takes priority if given
    if file is not None:
        try:
            # try to load a Normalizer pt file
            state_dict = torch.load(file)
        except RuntimeError:  # try to read an npz file
            # try to load an NPZ file
            values = np.load(file)
            state_dict = {}
            # legacy linref files:
            if "coeff" in values:
                state_dict["elementref"] = torch.tensor(values["coeff"])
            else:
                state_dict["elementref"] = torch.tensor(values["elementref"])

    if type == "linear":
        if "elementref" not in state_dict:
            raise RuntimeError("Unable to load linear element references!")
        references = LinearReference(element_references=state_dict["elementref"])
    else:
        raise ValueError(f"Invalid element references type={type}.")

    references.to(device)
    return references
