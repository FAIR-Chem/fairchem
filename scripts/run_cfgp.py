import os
import sys
import gpytorch

sys.path.insert(0, os.getcwd())
from ocpmodels.common.lbfgs import FullBatchLBFGS
from ocpmodels.models.gps import ExactGP
from ocpmodels.trainers.cfgp_trainer import CfgpTrainer
from ocpmodels.trainers.gpytorch_trainer import GPyTorchTrainer
from ocpmodels.trainers.simple_trainer import SimpleTrainer


if __name__ == "__main__":

    task = {
        "dataset": "gasdb",
        "description": "Binding energy regression on a dataset of DFT results for CO, H, N, O, and OH adsorption on various slabs.",
        "labels": ["binding energy"],
        "metric": "mae",
        "type": "regression",
    }

    model = {
        "name": "cgcnn",
        "atom_embedding_size": 64,
        "fc_feat_size": 128,
        "num_fc_layers": 4,
        "num_graph_conv_layers": 6,
    }

    dataset = {
        "src": "path/to/dataset",
        "train_size": 640,
        "val_size": 160,
        "test_size": 200,
    }

    optimizer = {
        "batch_size": 64,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [100, 150],
        "max_epochs": 50,
        "warmup_epochs": 10,
        "warmup_factor": 0.2,
    }

    cnn_trainer = SimpleTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="my-first-experiment",
    )

    gp_trainer = GPyTorchTrainer(
        Gp=ExactGP,
        Optimizer=FullBatchLBFGS,
        Likelihood=gpytorch.likelihoods.GaussianLikelihood,
        Loss=gpytorch.mlls.ExactMarginalLogLikelihood,
    )

    trainer = CfgpTrainer(cnn_trainer, gp_trainer)

    trainer.train()

    predictions, uncertainties = trainer.predict("path/to/test_dataset")
