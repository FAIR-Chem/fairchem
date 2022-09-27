"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform


@registry.register_dataset("oc22_lmdb")
class OC22LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(OC22LmdbDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        self.data2train = self.config.get("data2train", "all")
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                try:
                    length = pickle.loads(
                        self.envs[-1].begin().get("length".encode("ascii"))
                    )
                except TypeError:
                    length = self.envs[-1].stat()["entries"]
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

            if self.data2train != "all":
                txt_paths = sorted(self.path.glob("*.txt"))
                index = 0
                self.indices = []
                for txt_path in txt_paths:
                    lines = open(txt_path).read().splitlines()
                    for line in lines:
                        if self.data2train == "adslabs":
                            if "clean" not in line:
                                self.indices.append(index)
                        if self.data2train == "slabs":
                            if "clean" in line:
                                self.indices.append(index)
                        index += 1
                self.num_samples = len(self.indices)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.transform = transform
        self.lin_ref = self.oc20_ref = False
        self.train_total = self.config.get("total_energy", False)
        # only needed for oc20 datasets, oc22 is total by default
        if self.train_total:
            self.oc20_ref = pickle.load(open(config["oc20_ref"], "rb"))
        if self.config.get("lin_ref", False):
            coeff = np.load(self.config["lin_ref"], allow_pickle=True)["coeff"]
            self.lin_ref = torch.nn.Parameter(
                torch.tensor(coeff), requires_grad=False
            )
        self.subsample = self.config.get("subsample", False)

    def __len__(self):
        if self.subsample:
            return min(self.subsample, self.num_samples)
        return self.num_samples

    def __getitem__(self, idx):
        if self.data2train != "all":
            idx = self.indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)
        # make types consistent
        sid = data_object.sid
        if isinstance(sid, torch.Tensor):
            sid = sid.item()
            data_object.sid = sid
        if "fid" in data_object:
            fid = data_object.fid
            if isinstance(fid, torch.Tensor):
                fid = fid.item()
                data_object.fid = fid

        if hasattr(data_object, "y_relaxed"):
            attr = "y_relaxed"
        elif hasattr(data_object, "y"):
            attr = "y"
        # if targets are not available, test data is being used
        else:
            return data_object

        # convert s2ef energies to raw energies
        if attr == "y":
            # OC20 data
            if "oc22" not in data_object and self.train_total:
                randomid = f"random{sid}"
                data_object[attr] += self.oc20_ref[randomid]
                data_object.nads = 1
                data_object.oc22 = 0

        # convert is2re energies to raw energies
        else:
            if "oc22" not in data_object and self.train_total:
                randomid = f"random{sid}"
                data_object[attr] += self.oc20_ref[randomid]
                del data_object.force
                del data_object.y_init
                data_object.nads = 1
                data_object.oc22 = 0

        if self.lin_ref is not False:
            lin_energy = sum(self.lin_ref[data_object.atomic_numbers.long()])
            data_object[attr] -= lin_energy

        # to jointly train on oc22+oc20, need to delete these oc20-only attributes
        # ensure otf_graph=1 in your model configuration
        if "edge_index" in data_object:
            del data_object.edge_index
        if "cell_offsets" in data_object:
            del data_object.cell_offsets
        if "distances" in data_object:
            del data_object.distances

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
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()
