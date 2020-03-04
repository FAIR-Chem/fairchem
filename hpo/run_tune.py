import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, InMemoryDataset

import ray
from ray import tune

import sys
sys.path.append("/global/homes/b/bwood/machine_learning/ulissi_cnn/hpo/ocp_cgcnn/cgcnn")

from baselines.modules.normalizer import Normalizer
from baselines.models.cgcnn import CGCNN
from train_hpo import train, validate
from tune_hpo import TrainCGCNN

# TODO(Brandon) write a general tune.run() which will need to parse a config file of some sort
if __name__ == "__main__":
    ray.init()
    analysis = tune.run(
        TrainCGCNN,
        stop={"training_iteration": 60},
        resources_per_trial={
            "cpu": 5,
            "gpu": 1},
        num_samples=25,
        # checkpoint_at_end=True,
        # checkpoint_freq=3,
        config={"lr": tune.uniform(0.005, 0.1),
                "atom_embedding_size": tune.randint(46, 64),
                "batch_size": 80,
                "data_config": {"src": "/global/homes/b/bwood/machine_learning/ulissi_cnn/hpo/ocp_cgcnn/cgcnn/hpo/data_voronoi_H_4k_surface",
                                "train_size": 2560,
                                "val_size": 640,
                                "test_size": 800}},
        local_dir="/global/homes/b/bwood/machine_learning/hpo/results/H_4k_lr_atom_feat_25"
    )
    print("Best config is:", analysis.get_best_config(metric="validation_mae", mode="min", scope="last"))
