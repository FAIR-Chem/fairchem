"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

from fairchem.core.common.registry import registry
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.trainers import OCPTrainer

from .lr_scheduler import LRScheduler


@registry.register_trainer("equiformerv2_forces")
class EquiformerV2ForcesTrainer(OCPTrainer):
    # This trainer does a few things differently from the parent forces trainer:
    # - Support for cosine LR scheduler.
    # - When using the LR scheduler, it first converts the epochs into number of
    #   steps and then passes it to the scheduler. That way in the config
    #   everything can be specified in terms of epochs.
    def load_extras(self) -> None:
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config["optim"][
            "max_epochs"
        ]
        self.config["optim"]["scheduler_params"]["lr"] = self.config["optim"][
            "lr_initial"
        ]

        # convert epochs into number of steps
        if self.train_loader is None:
            logging.warning("Skipping scheduler setup. No training set found.")
            self.scheduler = None
        else:
            n_iter_per_epoch = len(self.train_loader)
            scheduler_params = self.config["optim"]["scheduler_params"]
            for k in scheduler_params:
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = int(
                            multiply(scheduler_params[k], n_iter_per_epoch)
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            int(x)
                            for x in multiply(scheduler_params[k], n_iter_per_epoch)
                        ]
            self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None
