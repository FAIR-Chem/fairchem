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
    TypeVar,
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


class UnsupportedDatasetError(ValueError):
    pass


class BaseDataset(Dataset[T_co], metaclass=ABCMeta):
    """Base Dataset class for all OCP datasets."""

    def __init__(self, config: dict):
        """Initialize

        Args:
            config (dict): dataset configuration
        """
        self.config = config
        self.paths = []

        if "src" in self.config:
            if isinstance(config["src"], str):
                self.paths = [Path(self.config["src"])]
            else:
                self.paths = tuple(Path(path) for path in sorted(config["src"]))

        self.lin_ref = None
        if self.config.get("lin_ref", False):
            lin_ref = torch.tensor(
                np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            )
            self.lin_ref = torch.nn.Parameter(lin_ref, requires_grad=False)

    def __len__(self) -> int:
        return self.num_samples

    def metadata_hasattr(self, attr) -> bool:
        return attr in self._metadata

    @cached_property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
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
            logging.warning(
                f"Could not find dataset metadata.npz files in '{self.paths}'"
            )
            return {}

        metadata = {
            field: np.concatenate([metadata[field] for metadata in metadata_npzs])
            for field in metadata_npzs[0]
        }

        assert np.issubdtype(
            metadata["natoms"].dtype, np.integer
        ), f"Metadata natoms must be an integer type! not {metadata['natoms'].dtype}"
        assert metadata["natoms"].shape[0] == len(
            self
        ), "Loaded metadata and dataset size mismatch."

        return metadata

    def get_metadata(self, attr, idx):
        if attr in self._metadata:
            metadata_attr = self._metadata[attr]
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None


class Subset(Subset_, BaseDataset):
    """A pytorch subset that also takes metadata if given."""

    def __init__(
        self,
        dataset: BaseDataset,
        indices: Sequence[int],
        metadata: dict[str, ArrayLike],
    ) -> None:
        super().__init__(dataset, indices)
        self.metadata = metadata
        self.indices = indices
        self.num_samples = len(indices)
        self.config = dataset.config

    @cached_property
    def _metadata(self) -> dict[str, ArrayLike]:
        return self.dataset._metadata

    def get_metadata(self, attr, idx):
        if isinstance(idx, list):
            return self.dataset.get_metadata(attr, [[self.indices[i] for i in idx]])
        return self.dataset.get_metadata(attr, self.indices[idx])


def create_dataset(config: dict[str, Any], split: str) -> Subset:
    """Create a dataset from a config dictionary

    Args:
        config (dict): dataset config dictionary
        split (str): name of split

    Returns:
        Subset: dataset subset class
    """
    # Initialize the dataset
    dataset_cls = registry.get_dataset_class(config.get("format", "lmdb"))
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls} is not a Dataset"

    # remove information about other splits, only keep specified split
    # this may only work with the mt config not main config
    current_split_config = config.copy()
    if "splits" in current_split_config:
        current_split_config.pop("splits")
        current_split_config.update(config["splits"][split])

    seed = current_split_config.get("seed", 0)
    if split != "train":
        seed += (
            1  # if we use same dataset for train / val , make sure its diff sampling
        )

    g = torch.Generator()
    g.manual_seed(seed)

    dataset = dataset_cls(current_split_config)

    # Get indices of the dataset
    indices = dataset.indices
    max_atoms = current_split_config.get("max_atoms", None)
    if max_atoms is not None:
        if not dataset.metadata_hasattr("natoms"):
            raise ValueError("Cannot use max_atoms without dataset metadata")
        indices = indices[dataset.get_metadata("natoms", indices) <= max_atoms]

    for subset_to in current_split_config.get("subset_to", []):
        if not dataset.metadata_hasattr(subset_to["metadata_key"]):
            raise ValueError(
                f"Cannot use {subset_to} without dataset metadata key {subset_to['metadata_key']}"
            )
        rhv = subset_to["rhv"]
        if isinstance(rhv, str):
            with open(rhv) as f:
                rhv = f.read().splitlines()
                rhv = [int(x) for x in rhv]
        if subset_to["op"] == "abs_le":
            indices = indices[
                np.abs(dataset.get_metadata(subset_to["metadata_key"], indices)) <= rhv
            ]
        elif subset_to["op"] == "in":
            indices = indices[
                np.isin(dataset.get_metadata(subset_to["metadata_key"], indices), rhv)
            ]

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
        indices = (
            indices
            if len(indices) == 1
            else indices[randperm(len(indices), generator=g)]
        )
    else:
        max_index = len(indices)
        indices = (
            indices
            if (no_shuffle or len(indices) == 1)
            else indices[randperm(len(indices), generator=g)]
        )

    if max_index > len(indices):
        msg = (
            f"Cannot take {max_index} data points from a dataset of only length {len(indices)}.\n"
            f"Make sure to set first_n or sample_n to a number =< the total samples in dataset."
        )
        if max_atoms is not None:
            msg = msg[:-1] + f"that are smaller than the given max_atoms {max_atoms}."
        raise ValueError(msg)

    indices = indices[:max_index]

    return Subset(dataset, indices, metadata=dataset._metadata)
