import glob
import json
import math
import os
import pickle
import random
from collections import defaultdict

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry


@registry.register_dataset("trajectory_lmdb")
class TrajectoryLmdbDataset(Dataset):
    def __init__(self, config, transform=None):
        super(TrajectoryLmdbDataset, self).__init__()

        world_size = distutils.get_world_size()
        rank = distutils.get_rank()

        self.config = config

        self.db_paths = glob.glob(
            os.path.join(self.config["src"], "") + "*lmdb"
        )
        self.txt_paths = glob.glob(
            os.path.join(self.config["src"], "") + "*txt"
        )
        assert len(self.db_paths) > 0, "No LMDBs found in {}".format(
            self.config["src"]
        )

        envs = [
            self.connect_db(self.db_paths[i])
            for i in range(len(self.db_paths))
        ]

        self._keys = [
            [
                j
                for j in range(
                    pickle.loads(envs[i].begin().get("length".encode("ascii")))
                )
            ]
            for i in range(len(self.db_paths))
        ]
        self._keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(self._keylens).tolist()

        self.transform = transform

        for i in range(len(envs)):
            envs[i].close()

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
        env = self.connect_db(self.db_paths[db_idx])
        datapoint_pickled = env.begin().get(
            f"{self._keys[db_idx][el_idx]}".encode("ascii")
        )
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
            map_size=1099511627776 / len(self.db_paths),
        )
        return env


# class TrajSampler(Sampler):
#     "Randomly samples batches of trajectories"
#
#     def __init__(self, data_source, traj_per_batch=5):
#         super().__init__(data_source)
#
#


class TrajSampler(Sampler):
    "Randomly samples batches of trajectories"

    def __init__(self, data_source, traj_per_batch=5):
        super().__init__(data_source)
        self.data_source = data_source
        self.system_samples = data_source._system_samples
        self.systemids = list(self.system_samples.keys())
        self.traj_batch = traj_per_batch

        # If running in distributed mode, only include a
        # subset of systems for each proces
        world_size = distutils.get_world_size()
        rank = distutils.get_rank()
        self.systemids = self.systemids[rank::world_size]
        self.num_samples = int(math.ceil(len(self.data_source) / world_size))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = []
        while len(indices) <= len(self):
            systemid = random.sample(self.systemids, 1)[0]
            system_indices = self.system_samples[systemid]
            if len(system_indices) < self.traj_batch:
                indices += system_indices
            else:
                indices += random.sample(system_indices, self.traj_batch)
        # trim excess samples
        indices = indices[: len(self)]
        return iter(indices)


def data_list_collater(data_list):
    n_neighbors = []
    for i, data in enumerate(data_list):
        n_index = data.edge_index[1, :]
        n_neighbors.append(n_index.shape[0])
    batch = Batch.from_data_list(data_list)
    batch.neighbors = torch.tensor(n_neighbors)
    return batch
