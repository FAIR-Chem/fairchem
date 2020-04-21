import datetime
import os
import warnings

import torch
import yaml
import torch.optim as optim

from ocpmodels.common.registry import registry
from ocpmodels.datasets import *
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.common.meter import Meter, mean_dist


@registry.register_trainer("dogss")
class DOGSSTrainer(BaseTrainer):
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
        logger="wandb",
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
                "logs_dir": os.path.join(run_dir, "logs", timestamp),
            },
        }

        os.makedirs(self.config["cmd"]["checkpoint_dir"])
        os.makedirs(self.config["cmd"]["results_dir"])
        os.makedirs(self.config["cmd"]["logs_dir"])

        self.is_debug = is_debug
        self.is_vis = is_vis
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.load()
        print(yaml.dump(self.config, default_flow_style=False))

        
    def load_criterion(self):
        self.criterion = mean_dist
    
    
    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), self.config["optim"]["lr_initial"]
        )
        
    def load_extras(self):
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.config["optim"]["lr_milestones"], gamma=self.config["optim"]["lr_gamma"]
        )

        # metrics.
        self.meter = Meter()   
        
    # Takes in a new data source and generates predictions on it.
#     def predict(self, src, batch_size=32):
#         print("### Generating predictions on {}.".format(src))

#         dataset_config = {"src": src}
#         dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
#             dataset_config
#         )
#         data_loader = dataset.get_full_dataloader(batch_size=batch_size)

#         self.model.eval()
#         predictions = []

#         for i, batch in enumerate(data_loader):
#             batch.to(self.device)
#             out, metrics = self._forward(batch)
#             if self.normalizers is not None and "target" in self.normalizers:
#                 out["output"] = self.normalizers["target"].denorm(
#                     out["output"]
#                 )
#             predictions.extend(out["output"].tolist())

#         return predictions
