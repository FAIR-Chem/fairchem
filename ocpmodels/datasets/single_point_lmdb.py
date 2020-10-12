"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pickle

import lmdb
from torch.utils.data import Dataset

from ocpmodels.common.registry import registry


@registry.register_dataset("single_point_lmdb")
class SinglePointLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing single point computations.
    Useful for Initial Structure to Relaxed Energy (IS2RE) task.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(SinglePointLmdbDataset, self).__init__()

        self.config = config

        self.db_path = self.config["src"]
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )

        env = self.connect_db(self.db_path)

        self._keys = [
            f"{j}".encode("ascii") for j in range(env.stat()["entries"])
        ]
        self.transform = transform

        env.close()

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # Return features.
        env = self.connect_db(self.db_path)
        datapoint_pickled = env.begin().get(self._keys[idx])
        data_object = pickle.loads(datapoint_pickled)
        data_object = (
            data_object
            if self.transform is None
            else self.transform(data_object)
        )
        env.close()

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            map_size=1099511627776 * 2,
        )
        return env
