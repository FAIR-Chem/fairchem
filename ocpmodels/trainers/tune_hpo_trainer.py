import datetime
import json
import os

import ray
import torch
from ray import tune
import copy

from ocpmodels.common.meter import Meter, mae, mae_ratio
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers import BaseTrainer


@registry.register_trainer("tune_hpo")
class TuneHPOTrainer(tune.Trainable, BaseTrainer):
    def _setup(self, config):
        self.load_config_from_dict_and_cmd()
        self.load_seed_from_config()
        self.load_task()
        self.load_model()
        self.load_criterion()
        self.load_optimizer()

    def _train(self):
        self.current_ip()
        tr_loss, tr_mae = self.train_for_one_epoch()
        va_loss, va_mae = self.validate(split="val")
        return {
            "training_loss": tr_loss,
            "training_mae": tr_mae,
            "validation_loss": va_loss,
            "validation_mae": va_mae,
        }

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
    
    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip

    def load_config_from_dict_and_cmd(self):
        # defaults.
        self.is_debug = False
        self.is_vis = False
        self.logger = None

        # Assumes this config has all the parameters,
        # even those from included files.
        #print(self.config)
        #self.config = config
        
        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # timestamps and directories
        """self.config["cmd"]["timestamp"] = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        if self.config["cmd"]["identifier"]:
            self.config["cmd"]["timestamp"] += "-{}".format(
                self.config["cmd"]["identifier"]
            )

        self.config["cmd"]["checkpoint_dir"] = os.path.join(
            "checkpoints", self.config["cmd"]["timestamp"]
        )
        self.config["cmd"]["results_dir"] = os.path.join(
            "results", self.config["cmd"]["timestamp"]
        )
        """
        # TODO(abhshkdz): Handle these parameters better. Maybe move to yaml.
        """
        os.makedirs(self.config["cmd"]["checkpoint_dir"])
        os.makedirs(self.config["cmd"]["results_dir"])

        # Dump config parameters
        json.dump(
            self.config,
            open(
                os.path.join(
                    self.config["cmd"]["checkpoint_dir"], "config.json"
                ),
                "w",
            ),
        )"""

    def train_for_one_epoch(self):
        self.model.train()
        meter = Meter()
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # Forward, loss, backward.
            out, metrics = self._forward(batch)
            loss = self._compute_loss(out, batch)
            self._backward(loss)

            # Update meter.
            meter_update_dict = {"loss": loss.item()}
            meter_update_dict.update(metrics)
            meter.update(meter_update_dict)

        return (
            float(meter.loss.global_avg),
            float(meter.meters["binding energy/mae"].global_avg),
        )

    def validate(self, split="val"):
        self.model.eval()
        meter = Meter()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in enumerate(loader):
            batch = batch.to(self.device)

            # Forward.
            out, metrics = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Update meter.
            meter_update_dict = {"loss": loss.item()}
            meter_update_dict.update(metrics)
            meter.update(meter_update_dict)

        return (
            float(meter.loss.global_avg),
            float(meter.meters["binding energy/mae"].global_avg),
        )
