import time
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ocpmodels.datasets import (
    TrajectoryDataset,
    TrajectoryFolderDataset,
    data_list_collater,
)
from ocpmodels.trainers import ForcesTrainer

if __name__ == "__main__":
    ########### BENCHMARKING DATALOADER
    # train_dataset_config = {
    #     "src": "data/data/2020_06_23_fbdft_1k",
    #     "traj_paths": "train.txt",
    #     "is_training": True,
    #     "mode": "uniform",
    #     "num_points": 50,
    #     "minp": 0.0,
    #     "maxp": 1.0,
    #     "normalize_labels": True,
    #     "target_mean": -343.19077959979006,
    #     "target_std": 229.0456275834696,
    #     "grad_target_mean": 0.0,
    #     "grad_target_std": 229.0456275834696,
    # }

    # val_dataset_config = {
    #     "src": "data/data/2020_06_23_fbdft_1k",
    #     "traj_paths": "val.txt",
    #     "is_training": False,
    #     "mode": "all"
    # }

    # dataset = TrajectoryFolderDataset(val_dataset_config)
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=5,
    #     shuffle=False,
    #     collate_fn=data_list_collater,
    #     num_workers=16,
    # )

    # start_time = time.time()
    # for i, batch in tqdm(enumerate(data_loader)):
    #     pass
    # print("time", time.time() - start_time)

    ########### BENCHMARKING OLDER TRAJECTORY LOADER
    # dataset_config = {
    #     "src": "/checkpoint/mshuaibi/06_23_2020_ocpdata_expt/val",
    #     "traj": "06_23_20_all_val.traj",
    #     "train_size": 42006,
    #     "val_size": 0,
    #     "test_size": 0,
    #     "normalize_labels": True,
    # }

    # print(time.time())
    # dataset = TrajectoryDataset(dataset_config)
    # data_loader = dataset.get_dataloaders(batch_size=1000, shuffle=False)
    # start_time = time.time()
    # for i, batch in tqdm(enumerate(data_loader)):
    #     pass
    # print("time", time.time() - start_time)

    ########### TRAINING SCRIPT
    task = {
        "dataset": "trajectory_folder",
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
        "src": "data/data/2020_06_23_fbdft_1k",
        "traj_paths": "train.txt",
        "is_training": True,  # energy, force annotations will be loaded.
        "mode": "uniform",
        "num_points": 32,
        "minp": 0.0,
        "maxp": 1.0,
        "normalize_labels": True,
        # normalization coefficients computed on the training set.
        # energy.
        "target_mean": 0.8607078617927244,
        "target_std": 26.649523952775112,
        # forces.
        "grad_target_mean": 0.0,
        "grad_target_std": 26.649523952775112,
    }

    val_dataset_config = {
        "src": "data/data/2020_06_23_fbdft_1k",
        "traj_paths": "val.txt",
        "is_training": True,
        "mode": "uniform",
        "num_points": 100,
    }

    optimizer = {
        "batch_size": 16,
        "num_workers": 16,
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
        dataset=[train_dataset_config, val_dataset_config],
        optimizer=optimizer,
        identifier="ocp-1k-schnet-t16-i32",
        print_every=1,
        is_debug=False,
        seed=1,
    )

    trainer.train()
