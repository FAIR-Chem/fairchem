"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABCMeta
from collections import namedtuple
from functools import cached_property
from pathlib import Path
from typing import TypeVar, Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset, Subset
from torch import randperm

from ocpmodels.common.registry import registry

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


def create_dataset(config: dict[str, Any], split: str):
    # TODO this might be better as a few funtions???
    # Initialize the dataset
    dataset_cls = registry.get_dataset_class(config["format"])
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls} is not a Dataset"

    # remove information about other splits, only keep specified split
    # this may only work with the mt config not main config
    current_split_config = config.copy()
    current_split_config.pop("splits")
    current_split_config.update(config["splits"][split])

    dataset = dataset_cls(current_split_config)

    # TODO return the dataset if max_atoms or others are not specified

    # Get indices of the dataset
    # TODO it might be good to move filter_indicies out of the BaseDataset class because
    # we have to parse the config here anyways
    max_atoms = current_split_config.get("max_atoms", None)
    if max_atoms is not None:
        # TODO add max_atoms argument to filtered_indices
        indices = dataset.filtered_indices(max_atoms)
    else:
        indices = dataset.indices

    # Apply dataset level transforms
    mutually_exclusive_arguments = ["sample_n", "first_n", "no_shuffle"]
    valid_arguments = (
        sum(
            argument in current_split_config
            for argument in mutually_exclusive_arguments
        )
        <= 1
    )  # this is true if at most of of the mutually exclusive arguments are set
    if not valid_arguments:
        raise IndexError(
            "sample_n, first_n, no_shuffle are mutually exclusive arguments"
        )
    if "first_n" in current_split_config:
        if len(dataset) < current_split_config["first_n"]:
            raise ValueError(
                f"Cannot take first {current_split_config['first_n']} from a dataset of only length {len(dataset)}"
            )
        indices = np.arange(current_split_config["first_n"])
        dataset = Subset(dataset, indices)
    elif (
        "no_shuffle" in current_split_config
        and current_split_config["no_shuffle"]
    ):
        return dataset
    else:  # shuffle by default , user can disable to optimize if they have confidence in dataset
        # shuffle all datasets by default to avoid biasing the sampling in concat dataset
        # TODO only shuffle is split is train
        indices = randperm(len(dataset)).tolist()
        if "sample_n" in current_split_config:
            if len(dataset) < current_split_config["sample_n"]:
                raise ValueError(
                    f"Cannot sample {current_split_config['sample_n']} from a dataset of only length {len(dataset)}"
                )
            indices = indices[: current_split_config["sample_n"]]
        dataset = Subset(dataset, indices)
    return dataset
