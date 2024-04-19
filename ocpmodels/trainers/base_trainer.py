"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import logging
import os
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from rich.console import Console
from rich.table import Table
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from tqdm import tqdm

from ocpmodels.common import dist_utils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.graph_transforms import RandomReflect, RandomRotate
from ocpmodels.common.registry import registry
from ocpmodels.common.timer import Times
from ocpmodels.common.utils import (
    JOB_ID,
    get_commit_hash,
    resolve,
    save_checkpoint,
)
from ocpmodels.datasets.data_transforms import FrameAveraging, get_transforms
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import EarlyStopper, LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(self, load=True, **kwargs):
        run_dir = kwargs["run_dir"]

        model_name = kwargs["model"].pop(
            "name", kwargs.get("model_name", "Unknown - base_trainer issue")
        )
        self.early_stopping_file = resolve(run_dir) / f"{str(uuid4())}.stop"
        kwargs["model"]["graph_rewiring"] = kwargs.get("graph_rewiring")

        self.config = {
            **kwargs,
            "model_name": model_name,
            "gpus": dist_utils.get_world_size() if not kwargs["cpu"] else 0,
            "checkpoint_dir": str(resolve(run_dir) / "checkpoints"),
            "results_dir": str(resolve(run_dir) / "results"),
            "logs_dir": str(resolve(run_dir) / "logs"),
            "early_stopping_file": str(self.early_stopping_file),
        }

        self.sigterm = False
        self.objective = None
        self.epoch = 0
        self.step = 0
        self.cpu = self.config["cpu"]
        self.task_name = self.config["task"].get("name", self.config.get("name"))
        assert self.task_name, "Specify task name (got {})".format(self.task_name)
        self.test_ri = self.config["test_ri"]
        self.is_debug = self.config["is_debug"]
        self.is_hpo = self.config["is_hpo"]
        self.eval_on_test = bool(self.config.get("eval_on_test"))
        self.silent = self.config["silent"]
        self.datasets = {}
        self.samplers = {}
        self.loaders = {}
        self.early_stopper = EarlyStopper(
            patience=self.config["optim"].get("es_patience") or 15,
            min_abs_change=self.config["optim"].get("es_min_abs_change") or 1e-5,
            min_lr=self.config["optim"].get("min_lr", -1),
            warmup_epochs=self.config["optim"].get("es_warmup_epochs") or -1,
        )
        self.config["commit"] = self.config.get("commit", get_commit_hash())

        if self.is_debug:
            del self.config["checkpoint_dir"]
            del self.config["results_dir"]
            del self.config["logdir"]
            del self.config["logs_dir"]
            del self.config["run_dir"]
            del self.config["early_stopping_file"]

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(self.device)
        # create directories from master rank only
        dist_utils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp.int()).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.timestamp_id = timestamp

        self.config["timestamp_id"] = self.timestamp_id

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp"] else None

        if JOB_ID and "folder" in self.config["slurm"]:
            self.config["slurm"]["job_id"] = JOB_ID
            self.config["slurm"]["folder"] = self.config["slurm"]["folder"].replace(
                "%j", self.config["slurm"]["job_id"]
            )

        self.config["dataset"] = kwargs["dataset"]
        deup_norm_key = [
            k for k in self.config["dataset"] if "deup" in k and "train" in k
        ]
        if deup_norm_key:
            self.train_dataset_name = deup_norm_key[0]
        else:
            self.train_dataset_name = "train"

        self.normalizer = kwargs["normalizer"]
        # This supports the legacy way of providing norm parameters in dataset
        if (
            self.config.get("dataset", None) is not None
            and kwargs["normalizer"] is None
        ):
            self.normalizer = self.config["dataset"][self.train_dataset_name]

        if not self.is_debug and dist_utils.is_master() and not self.is_hpo:
            os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["results_dir"], exist_ok=True)
            os.makedirs(self.config["logs_dir"], exist_ok=True)
            # TODO: do not create all three directory depending on mode
            # "Predict" -> result, "Train" -> checkpoint and logs.

        if self.is_hpo:
            # conditional import is necessary for checkpointing
            # from ray import tune

            from ocpmodels.common.hpo_utils import tune_reporter  # noqa: F401

            # sets the hpo checkpoint frequency
            # default is no checkpointing
            self.hpo_checkpoint_every = self.config["optim"].get("checkpoint_every", -1)

        if dist_utils.is_master() and not self.silent and not self.is_debug:
            print(f"\n🧰 Trainer config:\n{'-'*18}\n")
            print(yaml.dump(self.config), end="\n\n")
            print(
                f"\n\n🚦  Create {str(self.early_stopping_file)}",
                "to stop the training after the next validation\n",
            )
            (run_dir / f"config-{JOB_ID}.yaml").write_text(yaml.dump(self.config))

        # Here's the models whose edges are removed as a transform
        transform_models = [
            "depfaenet",
        ]
        if self.config["is_disconnected"]:
            print("\n\nHeads up: cat-ads edges being removed!")
        if self.config["model_name"] in transform_models:
            if not self.config["is_disconnected"]:
                print(
                    f"\n\nWhen using {self.config['model_name']},",
                    "the flag 'is_disconnected' should be used! The flag has been turned on.\n",
                )
                self.config["is_disconnected"] = True

        self.load()
        self.evaluator = Evaluator(
            task=self.task_name,
            model_regresses_forces=self.config["model"].get("regress_forces", ""),
        )

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if "cpu" not in str(self.device):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug and dist_utils.is_master() and not self.is_hpo:
            assert self.config["logger"] is not None, "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            self.logger = registry.get_logger_class(logger_name)(self.config)

    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=dist_utils.get_world_size(),
            rank=dist_utils.get_rank(),
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler):
        loader = DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )

        return loader

    def load_datasets(self):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model"].get("otf_graph", False),
        )

        transform = get_transforms(self.config)  # TODO: train/val/test behavior
        batch_size = self.config["optim"]["batch_size"]

        max_epochs = self.config["optim"].get("max_epochs", -1)
        max_steps = self.config["optim"].get("max_steps", -1)
        max_samples = self.config["optim"].get("max_samples", -1)

        for split, ds_conf in self.config["dataset"].items():
            if split == "default_val":
                continue

            if "deup" in split:
                self.datasets[split] = registry.get_dataset_class("deup_lmdb")(
                    self.config["dataset"],
                    split,
                    transform=transform,
                    silent=self.silent,
                )
            else:
                self.datasets[split] = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(
                    ds_conf,
                    transform=transform,
                    adsorbates=self.config.get("adsorbates"),
                    adsorbates_ref_dir=self.config.get("adsorbates_ref_dir"),
                    silent=self.silent,
                )

            if self.config["lowest_energy_only"]:
                with open(
                    "/network/scratch/a/alvaro.carbonero/lowest_energy.pkl", "rb"
                ) as fp:
                    good_indices = pickle.load(fp)
                good_indices = list(good_indices)

                self.real_dataset = self.datasets["train"]
                self.datasets["train"] = Subset(self.datasets["train"], good_indices)

            shuffle = False
            if "train" in split:
                shuffle = True
                n_train = len(self.datasets[split])

                if "fidelity_max_epochs" in self.config["optim"]:
                    self.config["optim"]["fidelity_max_steps"] = int(
                        np.ceil(
                            self.config["optim"]["fidelity_max_epochs"]
                            * (n_train / batch_size)
                        )
                    )
                    if not self.silent:
                        print(
                            "Setting fidelity_max_steps to {}".format(
                                self.config["optim"]["fidelity_max_steps"]
                            )
                        )

                if max_samples > 0:
                    if max_epochs > 0 and not self.silent:
                        print(
                            "\nWARNING: Both max_samples and max_epochs are set.",
                            "Using max_samples.",
                        )
                    if max_steps > 0 and not self.silent:
                        print(
                            "WARNING: Both max_samples and max_steps are set.",
                            "Using max_samples.\n",
                        )
                    self.config["optim"]["max_epochs"] = int(
                        np.ceil(max_samples / n_train)
                    )
                    self.config["optim"]["max_steps"] = int(
                        np.ceil(max_samples / batch_size)
                    )
                elif max_steps > 0:
                    if max_epochs > 0 and not self.silent:
                        print(
                            "\nWARNING: Both max_steps and max_epochs are set.",
                            "Using max_steps.\n",
                        )
                    self.config["optim"]["max_epochs"] = int(
                        np.ceil(max_steps / (n_train / batch_size))
                    )
                    if not self.silent:
                        print(
                            "Setting max_epochs to",
                            self.config["optim"]["max_epochs"],
                            f"from max_steps ({max_steps}),",
                            f"dataset length ({n_train}),",
                            f"and batch_size ({batch_size})\n",
                        )
                else:
                    self.config["optim"]["max_steps"] = int(
                        np.ceil(max_epochs * (n_train / batch_size))
                    )
                    if not self.silent:
                        print(
                            "Setting max_steps to ",
                            f"{self.config['optim']['max_steps']} from",
                            f"max_epochs ({max_epochs}), dataset length",
                            f"({n_train}), and batch_size ({batch_size})\n",
                        )

            self.samplers[split] = self.get_sampler(
                self.datasets[split], batch_size, shuffle=shuffle
            )
            self.loaders[split] = self.get_dataloader(
                self.datasets[split], self.samplers[split]
            )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
                if "hof_stats" in self.normalizer:
                    self.normalizers["target"].set_hof_rescales(
                        self.normalizer["hof_stats"]
                    )
            else:
                self.normalizers["target"] = Normalizer(
                    tensor=self.datasets["train"].data.y[
                        self.datasets["train"].__indices__
                    ],
                    device=self.device,
                )

    @abstractmethod
    def load_task(self):
        """
        Initialize task-specific information.
        Derived classes should implement this function.
        """
        pass

    def load_model(self):
        bond_feat_dim = None
        bond_feat_dim = self.config["model"].get("num_gaussians", 50)

        loader = list(self.loaders.values())[0] if self.loaders else None
        num_atoms = None
        if loader:
            sample = loader.dataset[0]
            if hasattr(sample, "x") and hasattr(sample.x, "shape"):
                num_atoms = sample.x.shape[-1]

        model_config = {
            **{
                "num_atoms": num_atoms,
                "bond_feat_dim": bond_feat_dim,
                "num_targets": self.num_targets,
                "task_name": self.task_name,
            },
            **self.config["model"],
            "model_name": self.config["model_name"],
        }

        self.model = registry.get_model_class(self.config["model_name"])(
            **model_config
        ).to(self.device)
        self.model.reset_parameters()
        self.model.set_deup_inference(False)

        if dist_utils.is_master() and not self.silent:
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        # if self.logger is not None:
        #     self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if dist_utils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device], output_device=self.device
            )

    def load_checkpoint(self, checkpoint_path, silent=False):
        if Path(checkpoint_path).is_dir():
            checkpoint_path = str(
                Path(checkpoint_path) / "checkpoints" / "best_checkpoint.pt"
            )
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        map_location = torch.device("cpu") if self.cpu else self.device
        if not silent:
            print(f"Loading checkpoint from: {checkpoint_path} onto {map_location}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        first_key = next(iter(checkpoint["state_dict"]))
        strict = "deup" not in self.config["config"]
        missing, unexpected = None, None
        if not dist_utils.initialized() and first_key.split(".")[1] == "module":
            # No need for OrderedDict since dictionaries are technically ordered
            # since Python 3.6 and officially ordered since Python 3.7
            new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            missing, unexpected = self.model.load_state_dict(new_dict, strict)
        elif dist_utils.initialized() and first_key.split(".")[1] != "module":
            new_dict = {f"module.{k}": v for k, v in checkpoint["state_dict"].items()}
            missing, unexpected = self.model.load_state_dict(new_dict, strict)
        else:
            missing, unexpected = self.model.load_state_dict(
                checkpoint["state_dict"], strict
            )

        if missing or unexpected:
            print("Warning: Model did not load correctly.")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

        if "optimizer" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if (
            "scheduler" in checkpoint
            and checkpoint["scheduler"] is not None
            and hasattr(self, "scheduler")
        ):
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if (
            checkpoint.get("warmup_scheduler") is not None
            and self.scheduler.warmup_scheduler is not None
        ):
            self.scheduler.warmup_scheduler.load_state_dict(
                checkpoint["warmup_scheduler"]
            )
        if (
            "ema" in checkpoint
            and checkpoint["ema"] is not None
            and hasattr(self, "ema")
        ):
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(checkpoint["normalizers"][key])
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

        if "config" in checkpoint:
            if "job_ids" in checkpoint["config"] and JOB_ID not in checkpoint["config"]:
                self.config["job_ids"] = checkpoint["config"]["job_ids"] + f", {JOB_ID}"

    def load_loss(self, reduction="mean"):
        self.loss_fn = {}
        self.loss_fn["energy"] = self.config["optim"].get("loss_energy", "mae")
        self.loss_fn["force"] = self.config["optim"].get("loss_force", "mae")
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ["l1", "mae"]:
                self.loss_fn[loss] = nn.L1Loss(reduction=reduction)
            elif loss_name == "mse":
                self.loss_fn[loss] = nn.MSELoss(reduction=reduction)
            elif loss_name == "l2mae":
                self.loss_fn[loss] = L2MAELoss(reduction=reduction)
            else:
                raise NotImplementedError(f"Unknown loss function name: {loss_name}")
            if dist_utils.initialized():
                self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")

        if optimizer.lower() == "amsgrad":
            optimizer = "Adam"
            if "optimizer_params" not in self.config["optim"]:
                self.config["optim"]["optimizer_params"] = {}
            self.config["optim"]["optimizer_params"]["amsgrad"] = True

        optimizer = getattr(optim, optimizer)

        if self.config["optim"].get("weight_decay", 0) > 0:
            # Do not regularize bias etc.
            params_decay = []
            params_no_decay = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        params_no_decay += [param]
                    elif "frequencies" in name:
                        params_no_decay += [param]
                    elif "bias" in name:
                        params_no_decay += [param]
                    else:
                        params_decay += [param]

            self.optimizer = optimizer(
                [
                    {"params": params_no_decay, "weight_decay": 0},
                    {
                        "params": params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

    def load_extras(self):
        self.scheduler = LRScheduler(
            self.optimizer,
            self.config["optim"],
            silent=self.silent,
        )
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if not self.is_debug and dist_utils.is_master():
            if training_state:
                ckpt_dict = {
                    "epoch": self.epoch,
                    "step": self.step,
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
                    "ema": self.ema.state_dict() if self.ema else None,
                    "amp": self.scaler.state_dict() if self.scaler else None,
                }
                if self.scheduler.warmup_scheduler is not None:
                    ckpt_dict[
                        "warmup_scheduler"
                    ] = self.scheduler.warmup_scheduler.state_dict()

                save_checkpoint(
                    ckpt_dict,
                    checkpoint_dir=self.config["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema:
                    self.ema.store()
                    self.ema.copy_to()
                save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict() if self.scaler else None,
                    },
                    checkpoint_dir=self.config["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
        dist_utils.synchronize()

    def save_hpo(self, epoch, step, metrics, checkpoint_every):
        # default is no checkpointing
        # checkpointing frequency can be adjusted by setting checkpoint_every in steps
        # to checkpoint every time results are communicated
        # to Ray Tune set checkpoint_every=1
        # from ray import tune

        if checkpoint_every != -1 and step % checkpoint_every == 0:
            with tune.checkpoint_dir(step=step) as checkpoint_dir:  # noqa: F821
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(self.save_state(epoch, step, metrics), path)

    def hpo_update(self, epoch, step, train_metrics, val_metrics, test_metrics=None):
        progress = {
            "steps": step,
            "epochs": epoch,
            "act_lr": self.optimizer.param_groups[0]["lr"],
        }
        # checkpointing must occur before reporter
        # default is no checkpointing
        self.save_hpo(
            epoch,
            step,
            val_metrics,
            self.hpo_checkpoint_every,
        )
        # report metrics to tune
        tune_reporter(  # noqa: F821
            iters=progress,
            train_metrics={k: train_metrics[k]["metric"] for k in self.metrics},
            val_metrics={k: val_metrics[k]["metric"] for k in val_metrics},
            test_metrics=test_metrics,
        )

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""
        pass

    def validate(
        self,
        split="val",
        disable_tqdm=True,
        debug_batches=-1,
        is_final=False,
        is_first=False,
    ):
        # Compute energy gradient (just for a metric)
        torch.set_grad_enabled(bool(self.config["model"].get("regress_forces", "")))

        if dist_utils.is_master() and not self.silent:
            print()
            logging.info(f"\n >>> 🧐 Evaluating on {split}.")
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator = Evaluator(
            task=self.task_name,
            model_regresses_forces=self.config["model"].get("regress_forces", ""),
        )
        metrics = {}
        desc = "device[rank={}]".format(dist_utils.get_rank())

        loader = self.loaders[split]
        times = Times(gpu=True)

        with times.next("validation_loop"):
            for i, batch in enumerate(tqdm(loader, desc=desc, disable=disable_tqdm)):
                if self.sigterm:
                    return "SIGTERM"

                if debug_batches > 0 and i == debug_batches:
                    break

                # Forward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    with times.next("val_forward", ignore=not is_first):
                        preds = self.model_forward(batch)
                    loss = self.compute_loss(preds, batch)

                # Compute metrics.
                metrics = self.compute_metrics(preds, batch, evaluator, metrics)
                for k, v in loss.items():
                    metrics = evaluator.update(k, v.item(), metrics)

        mean_val_times, std_val_times = times.prepare_for_logging(
            map_funcs={
                "val_forward": lambda x: x / self.config["optim"]["batch_size"],
            }
        )

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": dist_utils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": dist_utils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict["epoch"] = self.epoch
        log_dict[f"{split}_time"] = mean_val_times["validation_loop"]
        if is_first:
            log_dict["val_forward_time_mean"] = mean_val_times["val_forward"]
            log_dict["val_forward_time_std"] = std_val_times["val_forward"]

        if dist_utils.is_master() and not self.silent:
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            print(("\n  > ".join([""] + log_str))[1:])
            print()

        # Make plots.
        if self.logger is not None:
            log_dict = {f"{split}/{k}": v for k, v in log_dict.items()}
            if is_final:
                self.logger.log(
                    log_dict,
                    split="eval",
                )
            else:
                self.logger.log(
                    log_dict,
                    step=self.step,
                    split="val",
                )
        if self.ema:
            self.ema.restore()

        torch.set_grad_enabled(True)
        return metrics

    @abstractmethod
    def model_forward(self, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def compute_loss(self, out, batch_list):
        """Derived classes should implement this function."""

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss["total_loss"].backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None and (
                self.step % self.config.get("log_train_every", 1) == 0
            ):
                self.logger.log({"grad_norm": grad_norm}, step=self.step, split="train")
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

    def save_results(self, predictions, results_file, keys):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["results_dir"],
            f"{self.task_name}_{results_file}_{dist_utils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        dist_utils.synchronize()
        if dist_utils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["results_dir"],
                f"{self.task_name}_{results_file}.npz",
            )

            for i in range(dist_utils.get_world_size()):
                rank_path = os.path.join(
                    self.config["results_dir"],
                    f"{self.task_name}_{results_file}_{i}.npz",
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
                    gather_results[k] = np.concatenate(np.array(gather_results[k])[idx])
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(np.array(gather_results[k])[idx])[:-1]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

    def eval_all_splits(
        self, final=True, disable_tqdm=True, debug_batches=-1, epoch=-1, from_ckpt=None
    ):
        """Evaluate model on all four validation splits"""

        cumulated_time = 0
        cumulated_energy_mae = 0
        cumulated_forces_mae = 0
        metrics_dict = {}
        # store all non-train splits: all vals and test
        all_splits = [s for s in self.config["dataset"] if s.startswith("val")]

        if not self.silent:
            print()
            logging.info(f"Evaluating on {len(all_splits)} val splits.")

        # Load current best checkpoint for final evaluation
        if from_ckpt:
            self.load_checkpoint(checkpoint_path=from_ckpt)
        elif final and epoch != 0:
            checkpoint_path = os.path.join(
                self.config["checkpoint_dir"], "best_checkpoint.pt"
            )
            self.load_checkpoint(checkpoint_path=checkpoint_path)

        silent = self.silent
        self.silent = True
        metrics_names = None

        # evaluate on all splits
        for split in all_splits:
            start_time = time.time()
            self.metrics = self.validate(
                split=split,
                disable_tqdm=disable_tqdm,
                debug_batches=debug_batches,
                is_final=final,
            )

            if self.metrics == "SIGTERM":
                return "SIGTERM"

            metrics_dict[split] = self.metrics
            cumulated_energy_mae += self.metrics["energy_mae"]["metric"]
            if self.config["model"].get("regress_forces", False):
                cumulated_forces_mae += self.metrics["forces_mae"]["metric"]
            cumulated_time += time.time() - start_time
            if metrics_names is None:
                metrics_names = list(self.metrics.keys())

        self.silent = silent

        # Average metrics over all val splits
        metrics_dict["overall"] = {
            m: {
                "metric": sum([metrics_dict[s][m]["metric"] for s in all_splits])
                / len(all_splits)
            }
            for m in metrics_names
        }

        # Log specific metrics
        if final and self.config["logger"] == "wandb" and dist_utils.is_master():
            overall_energy_mae = cumulated_energy_mae / len(all_splits)
            self.logger.log({"Eval time": cumulated_time})
            self.objective = overall_energy_mae
            self.logger.log({"Eval time": cumulated_time})
            self.logger.log({"Overall MAE": overall_energy_mae})
            if self.config["model"].get("regress_forces", False):
                overall_forces_mae = cumulated_forces_mae / len(all_splits)
                self.logger.log({"Overall Forces MAE": overall_forces_mae})
                self.objective = (overall_energy_mae + overall_forces_mae) / 2
            self.logger.log({"Objective": self.objective})

        # Run on test split
        if final and "test" in self.config["dataset"] and self.eval_on_test:
            test_metrics = self.validate(
                split="test",
                disable_tqdm=disable_tqdm,
                debug_batches=debug_batches,
                is_final=final,
            )

            if test_metrics == "SIGTERM":
                return "SIGTERM"

            metrics_dict["test"] = test_metrics
            all_splits += ["test"]

        # Print results
        if not self.silent:
            if final:
                table = Table(title="Final results")
            elif epoch >= 0:
                table = Table(title=f"Results at epoch {epoch}")
            else:
                table = Table(title="Results")
            for c, col in enumerate(["Metric / Split"] + all_splits):
                table.add_column(col, justify="left" if c == 0 else "right")

            highlights = set()  # {"energy_mae", "forces_mae", "total_loss"}
            smn = sorted([m if "loss" not in m else f"z_{m}" for m in metrics_names])
            for metric in smn:
                metric = metric[2:] if metric.startswith("z_") else metric
                row = [metric] + [
                    f"{metrics_dict[split][metric]['metric']:.5f}"
                    for split in all_splits
                ]
                table.add_row(*row, style="on white" if metric in highlights else "")

            logging.info(f"eval_all_splits time: {time.time() - start_time:.2f}s")
            print()
            console = Console()
            console.print(table)
            print()
            print("\n• Trainer objective set to:", self.objective, end="\n\n")

    def rotate_graph(self, batch, rotation=None):
        """Rotate all graphs in a batch

        Args:
            batch (data.Batch): batch of graphs
            rotation (str, optional): type of rotation applied. Defaults to None.

        Returns:
            data.Batch: rotated batch
        """
        if isinstance(batch, list):
            batch = batch[0]

        # Sampling a random rotation within [-180, 180] for all axes.
        if rotation == "z":
            transform = RandomRotate([-180, 180], [2])
        elif rotation == "x":
            transform = RandomRotate([-180, 180], [0])
        elif rotation == "y":
            transform = RandomRotate([-180, 180], [1])
        else:
            transform = RandomRotate([-180, 180], [0, 1, 2])

        # Rotate graph
        batch_rotated, rot, inv_rot = transform(deepcopy(batch))
        assert not torch.allclose(batch.pos, batch_rotated.pos, atol=1e-05)

        # Recompute fa-pos for batch_rotated
        if hasattr(batch, "fa_pos"):
            delattr(batch_rotated, "fa_pos")  # delete it otherwise can't iterate
            delattr(batch_rotated, "fa_cell")  # delete it otherwise can't iterate
            delattr(batch_rotated, "fa_rot")  # delete it otherwise can't iterate

            g_list = batch_rotated.to_data_list()
            fa_transform = FrameAveraging(
                self.config["frame_averaging"], self.config["fa_method"]
            )
            for g in g_list:
                g = fa_transform(g)
            batch_rotated = Batch.from_data_list(g_list)
            if hasattr(batch, "neighbors"):
                batch_rotated.neighbors = batch.neighbors

        return {"batch_list": [batch_rotated], "rot": rot}

    def reflect_graph(self, batch, reflection=None):
        """Rotate all graphs in a batch

        Args:
            batch (data.Batch): batch of graphs
            rotation (str, optional): type of rotation applied. Defaults to None.

        Returns:
            data.Batch: rotated batch
        """
        if isinstance(batch, list):
            batch = batch[0]

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomReflect()

        # Reflect batch
        batch_reflected, rot, inv_rot = transform(deepcopy(batch))
        assert not torch.allclose(batch.pos, batch_reflected.pos, atol=1e-05)

        # Recompute fa-pos for batch_rotated
        if hasattr(batch, "fa_pos"):
            delattr(batch_reflected, "fa_pos")  # delete it otherwise can't iterate
            delattr(batch_reflected, "fa_cell")  # delete it otherwise can't iterate
            delattr(batch_reflected, "fa_rot")  # delete it otherwise can't iterate
            g_list = batch_reflected.to_data_list()
            fa_transform = FrameAveraging(
                self.config["frame_averaging"], self.config["fa_method"]
            )
            for g in g_list:
                g = fa_transform(g)
            batch_reflected = Batch.from_data_list(g_list)
            if hasattr(batch, "neighbors"):
                batch_reflected.neighbors = batch.neighbors

        return {"batch_list": [batch_reflected], "rot": rot}

    def scheduler_step(self, eval_every, metrics):
        if self.scheduler.scheduler_type == "ReduceLROnPlateau":
            if self.step % eval_every == 0:
                self.scheduler.step(
                    metrics=metrics,
                )
        else:
            self.scheduler.step()

    def handle_sigterm(self, signum, _):
        """
        Handle SIGTERM signal received.

        Args:
            signum (int): Signal number
        """
        if signum == 15 and not self.sigterm:
            print("\nHandling SIGTERM signal received.\n")
            self.sigterm = True

    def close_datasets(self):
        try:
            for ds in self.datasets.values():
                if hasattr(ds, "close_db") and callable(ds.close_db):
                    ds.close_db()
        except Exception as e:
            print("Error closing datasets: ", str(e))

    def measure_inference_time(self, loops=1):
        # keep grads if the model computes forces from energy
        enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(
            self.config["model"].get("regress_forces") == "from_energy"
        )
        self.model.eval()
        timer = Times(gpu=torch.cuda.is_available())

        # average inference over multiple loops
        for _ in range(loops):
            with timer.next("val_loop"):
                # iterate over default val set batches
                for b in self.loaders[self.config["dataset"]["default_val"]]:
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        # time forward pass
                        with timer.next("forward"):
                            _ = self.model_forward(b, mode="inference")

        # divide times by batch size
        mean, std = timer.prepare_for_logging(
            map_funcs={
                "forward": lambda t: self.config["optim"]["eval_batch_size"] / t,
            }
        )

        # log throughput to wandb as a summary metric
        if self.logger:
            if hasattr(self.logger, "run"):
                self.logger.run.summary["throughput_mean"] = mean["forward"]
                self.logger.run.summary["throughput_std"] = std["forward"]
                self.logger.run.summary["val_loop_time_mean"] = mean["val_loop"]
                self.logger.run.summary["val_loop_time_std"] = std["val_loop"]

        # print throughput to console
        if not self.silent:
            print(
                "Mean throughput:",
                f"{mean['forward']:.1f} +- {std['forward']:.1f} samples/s",
            )
            print(
                "Val loop time (around data-loader):",
                f"{mean['val_loop']:.3f} +- {std['val_loop']:.3f} s",
            )
        torch.set_grad_enabled(enabled)
