import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, InMemoryDataset
import ray
from ray import tune

import sys, os
sys.path.append("/global/homes/b/bwood/machine_learning/ulissi_cnn/hpo/ocp_cgcnn/cgcnn")
#from ray.tune.schedulers import ASHAScheduler

from baselines.modules.normalizer import Normalizer
from baselines.models.cgcnn import CGCNN
from dataloader import UlissiDataset, get_data_loaders
from train_hpo import train, validate
from tune_hpo import TrainCGCNN

# ray.init(address=args.ray_address)
# sched = ASHAScheduler(metric="mean_accuracy")
if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        TrainCGCNN,
        stop={"training_iteration": 4},
        resources_per_trial={
            "cpu": 1,
            "gpu": 1},
        num_samples=3,
        # checkpoint_at_end=True,
        # checkpoint_freq=3,
        config={"lr": tune.uniform(0.001, 0.1), "save_dir": "/global/homes/b/bwood/machine_learning/ulissi_cnn/hpo/ocp_cgcnn/cgcnn/hpo/voronoi_H_4k_data_splits"}, 
        local_dir="./results/hpo_cgcnn_test_single", 
        queue_trials=True
    )

    print("Best config is:", analysis.get_best_config(metric="validation_mae", mode="min", scope="last"))
    
    # "validation_mae": 0.35,
