import time
from itertools import chain

from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ocpmodels.datasets.traj_folder import TrajectoryFolderDataset
from ocpmodels.trainers import ForcesTrainer


# TODO(abhshkdz): move this to the dataset file.
def collater(data_list):
    # data_list is a list of lists of Data objects.
    # len(data_list) is no. of trajectories (batch_size in the DataLoader).
    # len(data_list[0]) is no. of points sampled from first trajectory.

    # flatten the list and make it into a torch_geometric.data.Batch.
    data_list = list(chain.from_iterable(data_list))
    batch = Batch.from_data_list(data_list)

    return batch


if __name__ == "__main__":
    # train_dataset_config = {
    #     "src": "data/data/2020_06_17_fbdft",
    #     "traj_paths": "train_debug.txt",
    #     "is_training": True,
    #     "mode": "uniform",
    #     "num_points": 10,
    #     "minp": 0.0,
    #     "maxp": 1.0,
    # }

    # val_dataset_config = {
    #     "src": "data/data/2020_06_17_fbdft",
    #     "traj_paths": "val_debug.txt",
    #     "is_training": True,
    #     "mode": "uniform",
    #     "num_points": 10,
    #     "minp": 0.0,
    #     "maxp": 1.0,
    # }

    # train_dataset = TrajectoryFolderDataset(train_dataset_config)

    task = {
        "dataset": "traj_folder",
        "description": "Regressing to binding energies for an MD trajectory of CO on Cu",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "grad_input": "atomic forces",
    }

    model = {
        "name": "schnet",
        "hidden_channels": 128,
        "num_filters": 128,
        "num_interactions": 3,
        "num_gaussians": 200,
        "cutoff": 6.0,
    }

    train_dataset_config = {
        "src": "data/data/2020_06_17_fbdft",
        "traj_paths": "train_debug.txt",
        "is_training": True,
        "mode": "uniform",
        "num_points": 50,
        "minp": 0.0,
        "maxp": 1.0,
        "normalize_labels": True,
        # mean, std on the training set.
        # energy.
        "target_mean": -343.19077959979006,
        "target_std": 229.0456275834696,
        # forces.
        # "grad_target_mean": [-7.15396186e-12, -1.36794091e-12,  1.22005523e-12],
        # "grad_target_std": [0.29951202, 0.37459846, 0.49785002],
        "grad_target_mean": 0.0,
        "grad_target_std": 229.0456275834696,
    }

    optimizer = {
        "batch_size": 5,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [100, 125],
        "max_epochs": 200,
        "warmup_epochs": 50,
        "warmup_factor": 0.2,
        "force_coefficient": 10,
    }

    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=train_dataset_config,
        optimizer=optimizer,
        identifier="schnet-debug",
        print_every=1,
        is_debug=True,
        seed=1,
    )

    trainer.train()
