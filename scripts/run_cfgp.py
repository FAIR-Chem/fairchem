import argparse
import os
import sys

import gpytorch
import torch.distributed as dist

from ocpmodels.common import distutils
from ocpmodels.common.lbfgs import FullBatchLBFGS
from ocpmodels.models.gps import ExactGP
from ocpmodels.trainers import CfgpTrainer, EnergyTrainer, GPyTorchTrainer


def main(args):
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    task = {
        "dataset": "single_point_lmdb",
        "description": "Relaxed state energy prediction from initial structure",
        "labels": ["relaxed_energy"],
        "metric": "mae",
        "type": "regression",
    }

    model = {
        "name": "cgcnn",
        "atom_embedding_size": 64,
        "fc_feat_size": 128,
        "num_fc_layers": 4,
        "num_graph_conv_layers": 6,
        "regress_forces": False,
    }

    dataset = {
        "src": "/home/jovyan/projects/aws-backup/debug_data/train_is/data.lmdb",
        "src": "/home/jovyan/projects/aws-backup/debug_data/train_is/data.lmdb",
    }

    optimizer = {
        "batch_size": 64,
        "num_workers": 16,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [100, 150],
        "max_epochs": 3,
        "warmup_epochs": 10,
        "warmup_factor": 0.2,
    }

    cnn_trainer = EnergyTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="my-first-experiment",
        local_rank=args.local_rank,
    )

    gp_trainer = GPyTorchTrainer(
        Gp=ExactGP,
        Optimizer=FullBatchLBFGS,
        Likelihood=gpytorch.likelihoods.GaussianLikelihood,
        Loss=gpytorch.mlls.ExactMarginalLogLikelihood,
    )

    trainer = CfgpTrainer(cnn_trainer, gp_trainer)
    trainer.train()

    trainer.predict(
        "/home/jovyan/projects/aws-backup/debug_data/train_is/data.lmdb"
    )

    if args.distributed:
        distutils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed", action="store_true", help="Run in DDP mode"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="GPU rank")
    args = parser.parse_args()
    main(args)
