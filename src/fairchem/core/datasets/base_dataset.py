"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from torch import randperm
from torch.utils.data import Dataset
from torch.utils.data import Subset as Subset_

from fairchem.core.common.registry import registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


T_co = TypeVar("T_co", covariant=True)


class DatasetMetadata(NamedTuple):
    natoms: ArrayLike | None = None


class UnsupportedDatasetError(ValueError):
    pass


@runtime_checkable
class DatasetWithSizes(Protocol):
    # metadata: DatasetMetadata

    def get_metadata(self, attr, idxs):
        """get metadata attr for the given idx or idxs"""


class Subset(Subset_, DatasetWithSizes):
    """A pytorch subset that also takes metadata if given."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        indices: Sequence[int],
        metadata: DatasetMetadata | None = None,
    ) -> None:
        super().__init__(dataset, indices)
        self.metadata = metadata
        self.indices = indices

    def get_metadata(self, attr, idx):
        if isinstance(idx, list):
            return getattr(self.dataset.metadata, attr)[[self.indices[i] for i in idx]]
        return getattr(self.dataset.metadata, attr)[self.indices[idx]]


class BaseDataset(Dataset[T_co], DatasetWithSizes, metaclass=ABCMeta):
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

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)

    def __len__(self) -> int:
        return self.num_samples

    @cached_property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    @cached_property
    def metadata(self) -> DatasetMetadata:
        # logic to read metadata file here
        metadata_npzs = []
        if self.config.get("metadata_path", None) is not None:
            metadata_npzs.append(
                np.load(self.config["metadata_path"], allow_pickle=True)
            )

        else:
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
                field: np.concatenate([metadata[field] for metadata in metadata_npzs])
                for field in DatasetMetadata._fields
            }
        )

        assert metadata.natoms.shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata

    def get_metadata(self, attr, idx):
        metadata_attr = getattr(self.metadata, attr)
        if isinstance(idx, list):
            return [metadata_attr[_idx] for _idx in idx]
        return metadata_attr[idx]


def create_dataset(config: dict[str, Any], split: str) -> Subset:
    """Create a dataset from a config dictionary

    Args:
        config (dict): dataset config dictionary
        split (str): name of split

    Returns:
        Subset: dataset subset class
    """
    # Initialize the dataset
    dataset_cls = registry.get_dataset_class(config["format"])
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls} is not a Dataset"

    # remove information about other splits, only keep specified split
    # this may only work with the mt config not main config
    current_split_config = config.copy()
    if "splits" in current_split_config:
        current_split_config.pop("splits")
        current_split_config.update(config["splits"][split])

    dataset = dataset_cls(current_split_config)
    # Get indices of the dataset
    indices = dataset.indices
    max_atoms = current_split_config.get("max_atoms", None)
    if max_atoms is not None:
        indices = indices[dataset.metadata.natoms[indices] <= max_atoms]

    # Apply dataset level transforms
    # TODO is no_shuffle mutually exclusive though? or what is the purpose of no_shuffle?
    first_n = current_split_config.get("first_n")
    sample_n = current_split_config.get("sample_n")
    no_shuffle = current_split_config.get("no_shuffle")
    # this is true if at most one of the mutually exclusive arguments are set
    if sum(arg is not None for arg in (first_n, sample_n, no_shuffle)) > 1:
        raise ValueError(
            "sample_n, first_n, no_shuffle are mutually exclusive arguments. Only one can be provided."
        )
    if first_n is not None:
        max_index = first_n
    elif sample_n is not None:
        # shuffle by default, user can disable to optimize if they have confidence in dataset
        # shuffle all datasets by default to avoid biasing the sampling in concat dataset
        # TODO only shuffle if split is train
        max_index = sample_n
        indices = indices[randperm(len(indices))]
    else:
        max_index = len(dataset)
        indices = indices if no_shuffle else indices[randperm(len(indices))]

    if max_index > len(indices):
        msg = (
            f"Cannot take {max_index} data points from a dataset of only length {len(indices)}.\n"
            f"Make sure to set first_n or sample_n to a number =< the total samples in dataset."
        )
        if max_atoms is not None:
            msg = msg[:-1] + f"that are smaller than the given max_atoms {max_atoms}."
        raise ValueError(msg)

    indices = indices[:max_index]

    return Subset(dataset, indices, metadata=dataset.metadata)
