import copy
import datetime
import json
import os
import sys

import ray
import torch
from ocpmodels.common.meter import Meter, mae, mae_ratio
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers import BaseTrainer
from ray import tune


@registry.register_trainer("tune_hpo")
# BaseTrainer
class TuneHPOTrainer(tune.Trainable):
    # TODO(Brandon): make _setup general to any trainer, once config dicts are standard
    def _setup(self, config):
        self.trainer = registry.get_trainer_class("simple")(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            identifier="",
            is_debug=True,
            seed=0,
            logger=None,
        )

        print("Device = {dev}".format(dev=self.trainer.device))
        # load() is part of the simple trainer but will need to added in the future
        # self.trainer.load()

    def _train(self):
        self.current_ip()
        metrics = self.trainer.train(max_epochs=1, return_metrics=True)
        return metrics

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.trainer.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.trainer.model.load_state_dict(torch.load(checkpoint_path))

    def current_ip(self):
        import socket

        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip
