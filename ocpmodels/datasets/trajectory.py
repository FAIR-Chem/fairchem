import os

import ase
import numpy as np
import torch
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import collate
from ocpmodels.datasets import BaseDataset
from ocpmodels.datasets.elemental_embeddings import EMBEDDINGS
from ocpmodels.datasets.gasdb import AtomicFeatureGenerator, GaussianDistance
from ocpmodels.preprocessing import AtomsToGraphs


@registry.register_dataset("trajectory")
class TrajectoryDataset(BaseDataset):
    def __init__(
        self, config, transform=None, pre_transform=None,
    ):
        super(BaseDataset, self).__init__(config, transform, pre_transform)

        self.config = config

        if (
            config.get("override_process", False)
            or os.path.isfile(self.processed_file_names[0]) is False
        ):
            self.process()
        else:
            self.data, self.slices = torch.load(self.processed_file_names[0])
            print(
                "### Loaded preprocessed data from:  {}".format(
                    self.processed_file_names
                )
            )

    @property
    def raw_file_names(self):
        return [os.path.join(self.config["src"], self.config["traj"])]

    @property
    def processed_file_names(self):
        os.makedirs(
            os.path.join(self.config["src"], "processed"), exist_ok=True
        )
        return [
            os.path.join(
                self.config["src"], "processed", self.config["traj"] + ".pt"
            )
        ]

    def process(self):
        print(
            "### Preprocessing atoms objects from:  {}".format(
                self.raw_file_names[0]
            )
        )
        traj = Trajectory(self.raw_file_names[0])
        a2g = AtomsToGraphs(
            max_neigh=self.config.get("max_neigh", 12),
            radius=self.config.get("radius", 6),
            dummy_distance=self.config.get("radius", 6) + 1,
            dummy_index=-1,
            r_energy=True,
            r_forces=True,
            r_distances=False,
        )

        data_list = []

        for atoms in tqdm(
            traj,
            desc="preprocessing atomic features",
            total=len(traj),
            unit="structure",
        ):
            data_list.append(a2g.convert(atoms))

        self.data, self.slices = collate(data_list)
        torch.save((self.data, self.slices), self.processed_file_names[0])

    def get_dataloaders(self, batch_size=None, shuffle=True, collate_fn=None):
        assert batch_size is not None
        assert self.train_size + self.val_size + self.test_size <= len(self)

        test_dataset = self[
            self.train_size
            + self.val_size : self.train_size
            + self.val_size
            + self.test_size
        ]
        train_val_dataset = self[: self.train_size + self.val_size].shuffle()

        train_loader = DataLoader(
            train_val_dataset[: self.train_size],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        if self.val_size == 0:
            val_loader = None
        else:
            val_loader = DataLoader(
                train_val_dataset[
                    self.train_size : self.train_size + self.val_size
                ],
                batch_size=batch_size,
                collate_fn=collate_fn,
            )

        if self.test_size == 0:
            test_loader = None
        else:
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, collate_fn=collate_fn
            )

        return train_loader, val_loader, test_loader
