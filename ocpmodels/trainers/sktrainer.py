import os
import warnings
import yaml
import datetime
import torch
from ocpmodels.common.registry import registry
from .base_trainer import BaseTrainer


@registry.register_trainer("sktrainer")
class SKTrainer(BaseTrainer):
    def __init__(self, task, model, dataset, optimizer, identifier,
                 run_dir=None, is_debug=False, is_vis=False,
                 print_every=100, seed=None, logger="wandb"):
        if run_dir is None:
            run_dir = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.config = {"task": task,
                       "dataset": dataset,
                       "model": model.pop("name"),
                       "model_attributes": model,
                       "optim": optimizer,
                       "logger": logger,
                       "cmd": {"identifier": identifier,
                               "print_every": print_every,
                               "seed": seed,
                               "timestamp": timestamp,
                               "checkpoint_dir": os.path.join(run_dir, "checkpoints", timestamp),
                               "results_dir": os.path.join(run_dir, "results", timestamp),
                               "logs_dir": os.path.join(run_dir, "logs", timestamp)}}

        os.makedirs(self.config["cmd"]["checkpoint_dir"])
        os.makedirs(self.config["cmd"]["results_dir"])
        os.makedirs(self.config["cmd"]["logs_dir"])

        self.is_debug = is_debug
        self.is_vis = is_vis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load()
        print(yaml.dump(self.config, default_flow_style=False))

    def predict(self):
        raise NotImplementedError
