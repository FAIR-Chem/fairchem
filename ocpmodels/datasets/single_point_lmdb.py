"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import errno
import os
import pickle
from pathlib import Path

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

        self.db_path = Path(self.config["src"])
        if not self.db_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT, "LMDB file not found", str(self.db_path)
            )

        self.metadata_path = self.db_path.parent / "metadata.npz"

        self.env = self.connect_db(self.db_path)

        self._keys = [
            f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
        ]
        self.transform = transform

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # Return features.
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data_object = pickle.loads(datapoint_pickled)
        data_object = (
            data_object
            if self.transform is None
            else self.transform(data_object)
        )

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        self.env.close()
