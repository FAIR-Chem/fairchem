import glob
import json
import os
import pickle

import lmdb
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry


@registry.register_dataset("trajectory_lmdb")
class TrajectoryLmdbDataset(Dataset):
    def __init__(self, config, transform=None):
        super(TrajectoryLmdbDataset, self).__init__()

        self.config = config

        self.db_paths = glob.glob(self.config["src"] + "*lmdb")
        envs = [
            lmdb.open(
                self.db_paths[i],
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                map_size=1099511627776 * 2,
            )
            for i in range(len(self.db_paths))
        ]
        self.db_txn = [envs[i].begin() for i in range(len(self.db_paths))]

        self._keys = [
            [f"{j}".encode("ascii") for j in range(envs[i].stat()["entries"])]
            for i in range(len(self.db_paths))
        ]
        self._keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(self._keylens).tolist()

        self.transform = transform

    def __len__(self):
        return sum(self._keylens)

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = 0
        for i in range(len(self._keylen_cumulative)):
            if self._keylen_cumulative[i] > idx:
                db_idx = i
                break

        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = self.db_txn[db_idx].get(self._keys[db_idx][el_idx])
        data_object = pickle.loads(datapoint_pickled)
        data_object = (
            data_object
            if self.transform is None
            else self.transform(data_object)
        )
        return data_object


def data_list_collater(data_list):
    batch = Batch.from_data_list(data_list)
    return batch
