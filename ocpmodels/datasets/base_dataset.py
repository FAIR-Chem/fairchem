"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABCMeta
from collections import namedtuple
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)
DatasetMetadata = namedtuple(
    "DatasetMetadata",
    [
        "natoms",
    ],
    defaults=[
        None,
    ],
)


class BaseDataset(Dataset[T_co], metaclass=ABCMeta):
    """Base Dataset class for all OCP datasets."""

    def __init__(self, config: dict):
        """Initialize

        Args:
            config (dict): dataset configuration
        """
        self.config = config

        if isinstance(config["src"], str):
            self.paths = [Path(self.config["src"])]
        else:
            self.paths = tuple(Path(path) for path in config["src"])

        if self.config.get("filter") is not None:
            max_natoms = self.config["filter"].get("max_natoms", None)
            self._data_filter = (
                lambda idx: self.metadata.natoms[idx] <= max_natoms
            )
        else:
            self._data_filter = lambda idx: True

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)

    def data_sizes(self, indices: ArrayLike) -> NDArray[int]:
        return self.metadata.natoms[indices]

    def __len__(self) -> int:
        return self.num_samples

    @cached_property
    def filtered_indices(self):
        return list(filter(self._data_filter, self.indices))

    @cached_property
    def indices(self):
        return list(range(self.num_samples))

    @cached_property
    def metadata(self) -> DatasetMetadata:
        # logic to read metadata file here
        metadata_npzs = []
        for path in self.paths:
            if path.is_file():
                metadata_file = path.parent / "metadata.npz"
            else:
                metadata_file = path / "metadata.npz"
            if metadata_file.is_file():
                metadata_npzs.append(np.load(metadata_file, allow_pickle=True))

        if len(metadata_npzs) == 0:
            raise ValueError(
                f"Could not find dataset metadata.npz files in '{self.paths}'"
            )

        metadata = DatasetMetadata(
            **{
                field: np.concatenate(
                    [metadata[field] for metadata in metadata_npzs]
                )
                for field in DatasetMetadata._fields
            }
        )

        assert metadata.natoms.shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata
