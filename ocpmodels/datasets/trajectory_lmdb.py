import json
import os
import pickle

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry


@registry.register_dataset("trajectory_lmdb")
class TrajectoryLmdbDataset(Dataset):
    def __init__(self, config):
        super(TrajectoryLmdbDataset, self).__init__()

        self.config = config
        env = lmdb.open(
            self.config["src"],
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            map_size=1099511627776 * 2,
        )
        self.db_txn = env.begin()
        self._keys = [
            f"{i}".encode("ascii") for i in range(env.stat()["entries"])
        ]

        self.inds = []

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        datapoint_pickled = self.db_txn.get(self._keys[idx])
        data_object = pickle.loads(datapoint_pickled)
        self.inds.append(idx)
        return data_object


def data_list_collater(data_list):
    batch = Batch.from_data_list(data_list)
    return batch
