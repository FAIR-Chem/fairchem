import datetime
import json
import os
import random
import time

import demjson
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
import yaml
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9

from cgcnn.common.logger import TensorboardLogger, WandBLogger
from cgcnn.common.meter import Meter, mae, mae_ratio
from cgcnn.common.registry import registry
from cgcnn.common.utils import (
    Complete,
    save_checkpoint,
    update_config,
    warmup_lr_lambda,
)
from cgcnn.datasets import ISO17, UlissigroupCO, XieGrossmanMatProj
from cgcnn.models import CGCNN
from cgcnn.modules.normalizer import Normalizer


class BaseTrainer:
    def __init__(self, args):
        # defaults.
        self.device = "cpu"
        self.is_debug = True
        # load config.
        self.load_config_from_yaml_and_cmd(args)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()
        self.load_criterion()
        self.load_optimizer()
        self.load_extras()

    def load_config_from_yaml_and_cmd(self, args):
        self.config = yaml.safe_load(open(args.config_yml, "r"))

        includes = self.config.get("includes", [])
        if not isinstance(includes, list):
            raise AttributeError(
                "Includes must be a list, {} provided".format(type(includes))
            )

        for include in includes:
            include_config = yaml.safe_load(open(include, "r"))
            self.config.update(include_config)

        self.config.pop("includes")

        if args.config_override:
            overrides = demjson.decode(args.config_override)
            self.config = update_config(self.config, overrides)

        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Are we just running sanity checks?
        self.is_debug = args.debug

        # timestamps and directories
        args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if args.identifier:
            args.timestamp += "-{}".format(args.identifier)

        args.checkpoint_dir = os.path.join("checkpoints", args.timestamp)
        args.results_dir = os.path.join("results", args.timestamp)
        args.logs_dir = os.path.join(
            "logs", self.config["logger"], args.timestamp
        )

        print(yaml.dump(self.config, default_flow_style=False))
        for arg in vars(args):
            print("{:<20}: {}".format(arg, getattr(args, arg)))

        # TODO(abhshkdz): Handle these parameters better. Maybe move to yaml.
        self.config["cmd"] = args.__dict__
        del args

        if not self.is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"])
            os.makedirs(self.config["cmd"]["results_dir"])
            os.makedirs(self.config["cmd"]["logs_dir"])

            # Dump config parameters
            json.dump(
                self.config,
                open(
                    os.path.join(
                        self.config["cmd"]["checkpoint_dir"], "config.json"
                    ),
                    "w",
                ),
            )

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"
            self.logger = registry.get_logger_class(self.config["logger"])(
                self.config
            )

    def load_task(self):
        # TODO(abhshkdz): move this out to a separate dataloader interface.
        print("### Loading {}".format(self.config["task"]["dataset"]))
        if self.config["task"]["dataset"] == "ulissigroup_co":
            dataset = UlissigroupCO(self.config["dataset"]["src"]).shuffle()
            num_targets = 1
        elif self.config["task"]["dataset"] == "xie_grossman_mat_proj":
            dataset = XieGrossmanMatProj(
                self.config["dataset"]["src"]
            ).shuffle()
            num_targets = 1
        elif self.config["task"]["dataset"] == "qm9":
            transform = T.Compose([Complete(), T.Distance(norm=False)])
            dataset = QM9(
                self.config["dataset"]["src"], transform=transform
            ).shuffle()
            num_targets = dataset.data.y.shape[-1]
            if (
                "label_index" in self.config["task"]
                and self.config["task"]["label_index"] is not False
            ):
                dataset.data.y = dataset.data.y[
                    :, int(self.config["task"]["label_index"])
                ]
                num_targets = 1
        elif self.config["task"]["dataset"] == "iso17":
            # TODO(abhshkdz): ISO17.test_other is currently broken.
            dataset = ISO17(self.config["dataset"]["src"])
            if self.config["dataset"]["test_fold"] == "test_within":
                self.config["dataset"]["test_size"] = 101000
            elif self.config["dataset"]["test_fold"] == "test_other":
                self.config["dataset"]["test_size"] = 130000
            else:
                raise NotImplementedError
            num_targets = 1
        else:
            raise NotImplementedError

        tr_sz, va_sz, te_sz = (
            self.config["dataset"]["train_size"],
            self.config["dataset"]["val_size"],
            self.config["dataset"]["test_size"],
        )

        assert len(dataset) > tr_sz + va_sz + te_sz

        train_dataset = dataset[:tr_sz]
        val_dataset = dataset[tr_sz : tr_sz + va_sz]
        test_dataset = dataset[tr_sz + va_sz : tr_sz + va_sz + te_sz]

        if self.config["task"]["dataset"] == "iso17":
            if self.config["dataset"]["test_fold"] == "test_within":
                test_dataset = dataset[tr_sz + va_sz : tr_sz + va_sz + te_sz]
            elif self.config["dataset"]["test_fold"] == "test_other":
                test_dataset = dataset[
                    tr_sz + va_sz + 101000 : tr_sz + va_sz + 101000 + te_sz
                ]
            else:
                raise NotImplementedError

        self.num_targets = num_targets
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["optim"]["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config["optim"]["batch_size"]
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config["optim"]["batch_size"]
        )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        self.normalizers["target"] = Normalizer(
            dataset.data.y[:tr_sz], self.device
        )

        # If we're computing gradients wrt input, compute mean, std of targets.
        if "grad_input" in self.config["task"]:
            self.normalizers["grad_target"] = Normalizer(
                dataset.data.forces[:tr_sz], self.device
            )

    def load_model(self):
        # Build model
        print("### Loading model")
        # TODO(abhshkdz): This currently only supports CGCNN.
        # TODO(abhshkdz): Remove dependency on self.train_loader.
        self.model = CGCNN(
            self.train_loader.dataset[0].x.shape[-1],
            self.train_loader.dataset[0].edge_attr.shape[-1],
            self.num_targets,
            **self.config["model"],
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(
            "### Loaded {} with {} parameters.".format(
                self.model.__class__.__name__, num_params
            )
        )

        if self.logger is not None:
            self.logger.watch(self.model)

    # TODO(abhshkdz): Rename function to something nicer.
    # TODO(abhshkdz): Support multiple loss functions.
    def load_criterion(self):
        self.criterion = nn.L1Loss()

    def load_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), self.config["optim"]["lr_initial"]
        )

    def load_extras(self):
        # learning rate scheduler.
        scheduler_lambda_fn = lambda x: warmup_lr_lambda(
            x, self.config["optim"]
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=scheduler_lambda_fn
        )

        # metrics.
        self.meter = Meter()

    def train(self):
        # TODO(abhshkdz): Timers for dataloading and forward pass.
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
                # TODO(abhshkdz): Checkpointing.
                if i % self.config["cmd"]["print_every"] == 0:
                    print(self.meter)

            self.scheduler.step()

            with torch.no_grad():
                self.validate(split="val", epoch=epoch)
                self.validate(split="test", epoch=epoch)

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))
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

        # Make plots.
        if self.logger is not None and epoch is not None:
            log_dict = meter.get_scalar_dict()
            log_dict.update({"epoch": epoch + 1})
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        print(meter)

    def _forward(self, batch):
        out, metrics = {}, {}

        # enable gradient wrt input.
        if "grad_input" in self.config["task"]:
            batch.x = batch.x.requires_grad_(True)

        # forward pass.
        output = self.model(batch)
        if batch.y.dim() == 1:
            output = output.view(-1)
        out["output"] = output

        force_output = None
        if "grad_input" in self.config["task"]:
            force_output = (
                self.config["task"]["grad_input_mult"]
                * torch.autograd.grad(
                    output,
                    batch.x,
                    # TODO(abhshkdz): check correctness. should this be `output`?
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            force_output = force_output[
                :,
                self.config["task"]["grad_input_start_idx"] : self.config[
                    "task"
                ]["grad_input_end_idx"],
            ]
            out["force_output"] = force_output

        # compute some metrics; don't compute loss here.
        # NOTE: only supports mean absolute error.
        metrics["mae"] = eval(self.config["task"]["metric"])(
            self.normalizers["target"].denorm(output).cpu(), batch.y.cpu()
        )
        if "grad_input" in self.config["task"]:
            metrics["grad_mae"] = eval(self.config["task"]["metric"])(
                self.normalizers["grad_target"].denorm(force_output).cpu(),
                batch.forces.cpu(),
            )

        return out, metrics

    def _compute_loss(self, out, batch):
        loss = []

        target_normed = self.normalizers["target"].norm(batch.y)
        loss.append(self.criterion(out["output"], target_normed))

        # TODO(abhshkdz): Test support for gradients wrt input.
        # TODO(abhshkdz): Make this general; remove dependence on `.forces`.
        if "grad_input" in self.config["task"]:
            grad_target_normed = self.normalizers["grad_target"].norm(
                batch.forces
            )
            loss.append(
                self.criterion(out["force_output"], grad_target_normed)
            )

        loss = sum(loss)
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # TODO(abhshkdz): Add support for gradient clipping.
        self.optimizer.step()
