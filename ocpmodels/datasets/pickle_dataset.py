"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
from pathlib import Path
from typing import TypeVar

import numpy as np

from ocpmodels.common.registry import registry
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from experimental.foundation_models.multi_task_dataloader.transforms.data_obejct import (
            DataTransforms,
            )
T_co = TypeVar("T_co", covariant=True)


class Env:
    def close(self):
        pass


@registry.register_dataset("pickle")
class PickleDataset(LmdbDataset[T_co]):
    r"""Dataset class to load from pickle file containing a list of elements
    config (dict): Dataset configuration (only use 'src' from this dictionary)
    transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None) -> None:
        # super(PickleDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        self._data_list = pickle.load(open(self.config["src"], "rb"))
        self.num_samples = len(self._data_list)
        self._keys = list(range(self.num_samples))

        # If specified, limit dataset to only a portion of the entire dataset
        # total_shards: defines total chunks to partition dataset
        # shard: defines dataset shard to make visible
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))
        self.env = Env()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> T_co:
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        data_object = self._data_list[idx]

        if self.key_mapping is not None:
            for _property in self.key_mapping:
                # catch for test data not containing labels
                if _property in data_object:
                    new_property = self.key_mapping[_property]
                    if new_property not in data_object:
                        data_object[new_property] = data_object[_property]
                        del data_object[_property]

        data_object = self.transforms(data_object)

        return data_object
