import os.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocpmodels.trainers import SimpleTrainer


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
        "src": "data/data/gasdb",
        "train_size": 800,
        "val_size": 100,
        "test_size": 100,
    }

    optimizer = {
        "batch_size": 10,
        "lr_gamma": 0.1,
        "lr_initial": 0.001,
        "lr_milestones": [100, 150],
        "max_epochs": 50,
        "warmup_epochs": 10,
        "warmup_factor": 0.2,
    }

    trainer = SimpleTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="my-first-experiment",
    )

    trainer.train()
    predictions = trainer.predict("data/data/gasdb")
