import datetime
import os
import warnings

import yaml

import torch
from ocpmodels.common.registry import registry
from ocpmodels.datasets import *
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("simple")
class SimpleTrainer(BaseTrainer):
    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        run_dir=None,
        is_debug=False,
        is_vis=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
    ):

        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if identifier:
            timestamp += "-{}".format(identifier)

        self.config = {
            "task": task,
            "dataset": dataset,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp": timestamp,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", timestamp
                ),
                "results_dir": os.path.join(run_dir, "results", timestamp),
                "logs_dir": os.path.join(run_dir, "logs", logger, timestamp),
            },
        }

        if not is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"])
            os.makedirs(self.config["cmd"]["results_dir"])
            os.makedirs(self.config["cmd"]["logs_dir"])

        self.is_debug = is_debug
        self.is_vis = is_vis
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    # Takes in a new data source and generates predictions on it.
    def predict(self, src, batch_size=32):
        print("### Generating predictions on {}.".format(src))

        dataset_config = {"src": src}
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            dataset_config
        )
        data_loader = dataset.get_full_dataloader(batch_size=batch_size)

        self.model.eval()
        predictions = []

        for i, batch in enumerate(data_loader):
            batch.to(self.device)
            out, metrics = self._forward(batch)
            if self.config["dataset"].get("normalize_labels", True):
                out["output"] = self.normalizers["target"].denorm(
                    out["output"]
                )
            predictions.extend(out["output"].tolist())

        return predictions

    def load_state(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file)["state_dict"]
        self.model.load_state_dict(state_dict)
