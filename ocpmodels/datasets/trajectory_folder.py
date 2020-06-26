import os
from itertools import chain

import numpy as np
import torch
from ase.io.trajectory import Trajectory
from torch_geometric.data import Batch, Data, Dataset

from ocpmodels.common.registry import registry
from ocpmodels.datasets.trajectory import TrajectoryFeatureGenerator
from ocpmodels.preprocessing import AtomsToGraphs


@registry.register_dataset("trajectory_folder")
class TrajectoryFolderDataset(Dataset):
    def __init__(self, config):
        super(TrajectoryFolderDataset, self).__init__(
            config["src"], transform=None, pre_transform=None
        )

        self.config = config

        with open(os.path.join(config["src"], config["traj_paths"]), "r") as f:
            self.raw_traj_files = f.read().splitlines()

        self.a2g = AtomsToGraphs(
            max_neigh=self.config.get("max_neigh", 12),
            radius=self.config.get("radius", 6),
            dummy_distance=self.config.get("dummy_distance", 7),
            dummy_index=self.config.get("dummy_index", -1),
            r_energy=self.config.get("is_training", False),
            r_forces=self.config.get("is_training", False),
            r_distances=False,
        )

    def len(self):
        return len(self.raw_traj_files)

    def get(self, idx):
        """
        Returns a list of torch_geometric.data.Data objects from the trajectory,
        each with atomic numbers, atomic positions, edge index, and optionally
        energies and forces during training.
        """
        raw_traj = Trajectory(self.raw_traj_files[idx])
        traj, _ = self.subsample_trajectory(
            raw_traj,
            self.config.get("mode", "all"),
            self.config.get("num_points", 1e6),
            self.config.get("minp", 0.0),
            self.config.get("maxp", 1.0),
        )

        # construct graph.
        data_list = self.a2g.convert_all(traj, disable_tqdm=True)

        return data_list

    def subsample_trajectory(
        self, traj, mode="all", num_points=1e6, minp=0.0, maxp=1.0
    ):
        """
        Returns subsampled trajectory and corresponding indices, depending on
        subsampling config parameters.

        We first subselect part of the trajectory based on `minp` and `maxp`:
        traj = traj[minp * len(traj) : maxp * len(traj)]
        `minp` and `maxp` to be specified as fractions (of trajectory length).

        1) If mode == "all", returns the entire traj. Ignores num_points.
        2) If mode == "uniform", uniformly samples num_points from the traj. (In
           case the traj is shorter than num_points, will return all points.)

        Common intended use cases:

        For training / validation on non-overlapping sets of full trajectories,
        pass mode = "all" and use the traj_paths txt file to specify splits.

        For training / validation on non-overlapping parts of the same traj,
        use different `minp` and `maxp` for training and val splits, with mode
        == "uniform" or mode == "all".
        """
        assert mode in ["all", "uniform"]
        assert minp >= 0.0 and maxp >= 0.0

        inds = list(range(len(traj)))

        traj = traj[int(minp * len(traj)) : int(maxp * len(traj))]
        inds = inds[int(minp * len(inds)) : int(maxp * len(inds))]

        if mode == "all":
            return traj, inds

        if mode == "uniform":
            sub_inds = np.random.choice(
                len(traj), min(num_points, len(traj)), replace=False
            )
            traj = [traj[i] for i in sub_inds]
            inds = [inds[i] for i in sub_inds]
            return traj, inds


def data_list_collater(data_list):
    # flatten the list and make it into a torch_geometric.data.Batch.
    data_list = list(chain.from_iterable(data_list))
    batch = Batch.from_data_list(data_list)
    return batch
