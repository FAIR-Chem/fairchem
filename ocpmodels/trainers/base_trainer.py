"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import json
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

import ocpmodels
from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.logger import TensorboardLogger, WandBLogger
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    plot_histogram,
    save_checkpoint,
    warmup_lr_lambda,
)
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
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
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        name="base_trainer",
    ):
        self.name = name
        self.cpu = cpu
        self.start_step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = local_rank
        else:
            self.device = "cpu"
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available
        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
            self.device
        )
        # create directories from master rank only
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        if identifier:
            timestamp += "-{}".format(identifier)
        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        ocpmodels.__path__[0],
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        self.config = {
            "task": task,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp": timestamp,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", timestamp
                ),
                "results_dir": os.path.join(run_dir, "results", timestamp),
                "logs_dir": os.path.join(run_dir, "logs", logger, timestamp),
            },
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if isinstance(dataset, list):
            self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        else:
            self.config["dataset"] = dataset

        if not is_debug and distutils.is_master() and not is_hpo:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_vis = is_vis
        self.is_hpo = is_hpo

        if distutils.is_master():
            print(yaml.dump(self.config, default_flow_style=False))
        self.load()

        self.evaluator = Evaluator(task=name)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()
        self.load_criterion()
        self.load_optimizer()
        self.load_extras()

    # Note: this function is now deprecated. We build config outside of trainer.
    # See build_config in ocpmodels.common.utils.py.
    def load_config_from_yaml_and_cmd(self, args):
        self.config = build_config(args)

        # AMP Scaler
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.config["amp"] else None
        )

        # device
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and not self.cpu) else "cpu"
        )

        # Are we just running sanity checks?
        self.is_debug = args.debug
        self.is_vis = args.vis

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
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

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
        if not self.is_debug and distutils.is_master() and not self.is_hpo:
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"
            self.logger = registry.get_logger_class(self.config["logger"])(
                self.config
            )

    @abstractmethod
    def load_task(self):
        """ Derived classes should implement this function."""

    def load_model(self):
        # Build model
        if distutils.is_master():
            print("### Loading model: {}".format(self.config["model"]))

        # TODO(abhshkdz): Eventually move towards computing features on-the-fly
        # and remove dependence from `.edge_attr`.
        bond_feat_dim = None
        if self.config["task"]["dataset"] in [
            "trajectory_lmdb",
            "single_point_lmdb",
        ]:
            bond_feat_dim = self.config["model_attributes"].get(
                "num_gaussians", 50
            )
        else:
            raise NotImplementedError

        self.model = registry.get_model_class(self.config["model"])(
            self.train_loader.dataset[0].x.shape[-1]
            if hasattr(self.train_loader.dataset[0], "x")
            and self.train_loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            print(
                "### Loaded {} with {} parameters.".format(
                    self.model.__class__.__name__, self.model.num_params
                )
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_pretrained(self, checkpoint_path=None, ddp_to_dp=False):
        if checkpoint_path is None or os.path.isfile(checkpoint_path) is False:
            print(f"Checkpoint: {checkpoint_path} not found!")
            return False

        print("### Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.start_step = checkpoint.get("step", 0)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        if ddp_to_dp:
            new_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])
        return True

    # TODO(abhshkdz): Rename function to something nicer.
    # TODO(abhshkdz): Support multiple loss functions.
    def load_criterion(self):
        self.criterion = self.config["optim"].get("criterion", nn.L1Loss())

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        self.optimizer = optimizer(
            params=self.model.parameters(),
            lr=self.config["optim"]["lr_initial"],
            **self.config["optim"].get("optimizer_params", {}),
        )

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        # metrics.
        self.meter = Meter(split="train")

    def save(self, epoch, step, metrics):
        if not self.is_debug and distutils.is_master() and not self.is_hpo:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "step": step,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.scheduler.state_dict()
                    if self.scheduler.scheduler_type != "Null"
                    else None,
                    "normalizers": {
                        key: value.state_dict()
                        for key, value in self.normalizers.items()
                    },
                    "config": self.config,
                    "val_metrics": metrics,
                    "amp": self.scaler.state_dict() if self.scaler else None,
                },
                self.config["cmd"]["checkpoint_dir"],
            )

    @abstractmethod
    def train(self):
        """ Derived classes should implement this function."""

    @torch.no_grad()
    def validate(self, split="val", epoch=None, disable_tqdm=False):
        if distutils.is_master():
            print("### Evaluating on {}.".format(split))
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": epoch + 1})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            print(", ".join(log_str))

        # Make plots.
        if self.logger is not None and epoch is not None:
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        return metrics

    @abstractmethod
    def _forward(self, batch_list):
        """ Derived classes should implement this function."""

    @abstractmethod
    def _compute_loss(self, out, batch_list):
        """ Derived classes should implement this function."""

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # TODO(abhshkdz): Add support for gradient clipping.
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def save_results(self, predictions, results_file, keys):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                f"{self.name}_{results_file}.npz",
            )

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(gather_results["ids"])[idx]
            for k in keys:
                if k == "forces":
                    gather_results[k] = np.concatenate(
                        np.array(gather_results[k])[idx]
                    )
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(
                        np.array(gather_results[k])[idx]
                    )[:-1]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            print(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)
