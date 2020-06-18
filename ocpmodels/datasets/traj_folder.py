import os

import numpy as np
import torch
from ase.io.trajectory import Trajectory
from torch_geometric.data import Data, Dataset

from ocpmodels.common.registry import registry
from ocpmodels.datasets.co_cu_md import TrajectoryFeatureGenerator


@registry.register_dataset("traj_folder")
class TrajectoryFolderDataset(Dataset):
    def __init__(self, config):
        super(TrajectoryFolderDataset, self).__init__(
            config["src"], transform=None, pre_transform=None
        )

        self.config = config
        with open(os.path.join(config["src"], config["traj_paths"]), "r") as f:
            self.raw_traj_files = f.read().splitlines()

    def len(self):
        return len(self.raw_traj_files)

    def get(self, idx):
        """
        Returns a list of torch_geometric.data.Data objects from the trajectory,
        each with atomic numbers, atomic positions, edge index, and optionally
        energies and forces during training.
        """
        raw_traj = Trajectory(self.raw_traj_files[idx])

        # TODO(abhshkdz): this is where sampling logic goes.
        # Add parameters for how many points to sample from each trajectory,
        # and whether to sample uniformly or bias towards beginning / end.

        # To be removed: uniformly sample 10 points from this trajectory.
        inds = np.random.choice(
            len(raw_traj), min(10, len(raw_traj)), replace=False
        )

        # Now prune the trajectory based on the above indices.
        traj = [raw_traj[i] for i in inds]
        feature_generator = TrajectoryFeatureGenerator(traj)

        # Extract torch_geometric.data.Data objects for each step.
        data_list = []
        for i, (_, _, index, positions, atomic_numbers) in enumerate(
            feature_generator
        ):
            edge_index = [[], []]

            # edge index.
            for j in range(index.shape[0]):
                for k in range(index.shape[1]):
                    edge_index[0].append(j)
                    edge_index[1].append(index[j, k])
            edge_index = torch.LongTensor(edge_index)

            # energy, forces.
            # TODO(abhshkdz): currently throws an error when is_training=False.
            p_energy, force = None, None
            if self.config["is_training"]:
                p_energy = traj[i].get_potential_energy(apply_constraint=False)
                force = traj[i].get_forces(apply_constraint=False)

            data_list.append(
                Data(
                    atomic_numbers=atomic_numbers,
                    pos=positions,
                    natoms=torch.tensor([positions.shape[0]]),
                    edge_index=edge_index,
                    y=p_energy,
                    force=torch.tensor(force),
                )
            )

        return data_list
