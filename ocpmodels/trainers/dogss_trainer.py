import datetime
import os
import warnings

import torch
import torch.optim as optim
import yaml
import numpy as np

from ocpmodels.common.meter import Meter, mean_l2_distance
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import save_checkpoint
from ocpmodels.datasets import DOGSS
from ocpmodels.trainers.base_trainer import BaseTrainer


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
        
        initial_train_loss = self.get_initial_loss(self.train_loader)
        initial_val_loss = self.get_initial_loss(self.val_loader)
        initial_test_loss = self.get_initial_loss(self.test_loader)
        print(" initial train loss: %f\n" %initial_train_loss,
              "initial val loss: %f\n" %initial_val_loss,
              "initial test loss: %f\n" %initial_test_loss,
             )
        
    def load_criterion(self):
        self.criterion = mean_l2_distance

    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), self.config["optim"]["lr_initial"], weight_decay=self.config["optim"]["weight_decay"]
        )

    def load_extras(self):
        # learning rate scheduler.
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config["optim"]["lr_milestones"],
            gamma=self.config["optim"]["lr_gamma"],
        )

        # metrics.
        self.meter = Meter()

    def train(self):
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)

                # Forward, loss, backward.
                out, metrics = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Update meter.
                meter_update_dict = {
                    "epoch": epoch + (i + 1) / len(self.train_loader),
                    "loss": loss.item(),
                }
                meter_update_dict.update(metrics)
                self.meter.update(meter_update_dict)

                # Make plots.
                if self.logger is not None:
                    self.logger.log(
                        meter_update_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                # Print metrics.
                if i % self.config["cmd"]["print_every"] == 0:
                    print(self.meter)

            self.scheduler.step()

            self.validate(split="val", epoch=epoch)
            self.validate(split="test", epoch=epoch)

            if not self.is_debug:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                    },
                    self.config["cmd"]["checkpoint_dir"],
                )
                
    def get_initial_loss(self, dataset):
        distances = []
        for data in dataset:
            free_atom_idx = np.where(data.fixed_base.cpu() == 0)[0]
            atom_pos = data.atom_pos[free_atom_idx]
            y = data.y
            dist = torch.sqrt(torch.sum((atom_pos-y)**2, dim=1))
            distances.append(dist)
        mae = torch.mean(torch.cat(distances))
        return mae