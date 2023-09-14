"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.trainers import ForcesTrainer

from .lr_scheduler import LRScheduler


def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    name_no_wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            name.endswith(".bias")
            or name.endswith(".affine_weight")
            or name.endswith(".affine_bias")
            or name.endswith(".mean_shift")
            or "bias." in name
            or any(name.endswith(skip_name) for skip_name in skip_list)
        ):
            no_decay.append(param)
            name_no_wd.append(name)
        else:
            decay.append(param)
    name_no_wd.sort()
    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return params, name_no_wd


@registry.register_trainer("equiformerv2_forces")
class EquiformerV2ForcesTrainer(ForcesTrainer):
    # This trainer does a few things differently from the parent forces trainer:
    # - Different way of setting up model parameters with no weight decay.
    # - Support for cosine LR scheduler.
    # - When using the LR scheduler, it first converts the epochs into number of
    #   steps and then passes it to the scheduler. That way in the config
    #   everything can be specified in terms of epochs.
    def load_model(self):
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        self.model = registry.get_model_class(self.config["model"])(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        # for no weight decay
        self.model_params_no_wd = {}
        if hasattr(self.model, "no_weight_decay"):
            self.model_params_no_wd = self.model.no_weight_decay()

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)
        optimizer_params = self.config["optim"]["optimizer_params"]
        weight_decay = optimizer_params["weight_decay"]

        parameters, name_no_wd = add_weight_decay(
            self.model, weight_decay, self.model_params_no_wd
        )
        logging.info("Parameters without weight decay:")
        logging.info(name_no_wd)

        self.optimizer = optimizer(
            parameters,
            lr=self.config["optim"]["lr_initial"],
            **optimizer_params,
        )

    def load_extras(self):
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config[
            "optim"
        ]["max_epochs"]
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
            for k in scheduler_params.keys():
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = int(
                            multiply(scheduler_params[k], n_iter_per_epoch)
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            int(x)
                            for x in multiply(
                                scheduler_params[k], n_iter_per_epoch
                            )
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
