"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.registry import registry
from ocpmodels.common.graph_transforms import RandomReflect, RandomRotate
from ocpmodels.common.utils import get_commit_hash, save_checkpoint, OCP_TASKS
from ocpmodels.datasets.data_transforms import FrameAveraging, get_transforms
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(self, **kwargs):

        run_dir = kwargs["run_dir"]
        model_name = kwargs["model"].pop("name")
        kwargs["model"]["graph_rewiring"] = kwargs.get("graph_rewiring")

        self.config = {
            **kwargs,
            "model_name": model_name,
            "gpus": distutils.get_world_size() if not kwargs["cpu"] else 0,
            "commit": get_commit_hash(),
            "checkpoint_dir": str(Path(run_dir) / "checkpoints"),
            "results_dir": str(Path(run_dir) / "results"),
            "logs_dir": str(Path(run_dir) / "logs"),
        }

        self.epoch = 0
        self.step = 0
        self.cpu = self.config["cpu"]
        self.task_name = self.config.get("task_name", self.config.get("name"))
        assert self.task_name, "Specify task name (got {})".format(self.task_name)
        self.test_ri = self.config["test_ri"]
        self.is_debug = self.config["is_debug"]
        self.is_hpo = self.config["is_hpo"]
        self.silent = self.config["silent"]

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{self.config['local_rank']}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(self.device)
        # create directories from master rank only
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(timestamp.int()).strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.timestamp_id = timestamp

        self.config["timestamp_id"] = self.timestamp_id

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config["amp"] else None

        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"]["folder"].replace(
                "%j", self.config["slurm"]["job_id"]
            )

        if isinstance(kwargs["dataset"], list):
            if len(kwargs["dataset"]) > 0:
                self.config["dataset"] = kwargs["dataset"][0]
            if len(kwargs["dataset"]) > 1:
                self.config["val_dataset"] = kwargs["dataset"][1]
            if len(kwargs["dataset"]) > 2:
                self.config["test_dataset"] = kwargs["dataset"][2]
        elif isinstance(kwargs["dataset"], dict):
            self.config["dataset"] = kwargs["dataset"].get("train", None)
            self.config["val_dataset"] = kwargs["dataset"].get("val", None)
            self.config["test_dataset"] = kwargs["dataset"].get("test", None)
        else:
            self.config["dataset"] = kwargs["dataset"]

        self.normalizer = kwargs["normalizer"]
        # This supports the legacy way of providing norm parameters in dataset
        if (
            self.config.get("dataset", None) is not None
            and kwargs["normalizer"] is None
        ):
            self.normalizer = self.config["dataset"]

        if not self.is_debug and distutils.is_master() and not self.is_hpo:
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

        if distutils.is_master() and not self.silent:
            print(yaml.dump(self.config, default_flow_style=False))
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug and distutils.is_master() and not self.is_hpo:
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
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
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

        self.train_loader = self.val_loader = self.test_loader = None

        transform = get_transforms(self.config)  # TODO: train/val/test behavior

        if self.config.get("dataset", None):
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"], transform=transform)
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            if self.config.get("val_dataset", None):
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(
                    self.config["val_dataset"],
                    transform=transform,
                )
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            if self.config.get("test_dataset", None):
                self.test_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(
                    self.config["test_dataset"],
                    transform=transform,
                )
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
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
            else:
                self.normalizers["target"] = Normalizer(
                    tensor=self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
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
        # Build model
        if distutils.is_master() and not self.silent:
            logging.info(
                f"Loading model {self.config['model_name']}: {self.config['model']}"
            )

        bond_feat_dim = None
        bond_feat_dim = self.config["model"].get("num_gaussians", 50)

        loader = self.train_loader or self.val_loader or self.test_loader
        num_atoms = None
        if loader:
            sample = loader.dataset[0]
            if hasattr(sample, "x") and hasattr(sample.x, "shape"):
                num_atoms = sample.x.shape[-1]

        self.model = registry.get_model_class(self.config["model_name"])(
            num_atoms=num_atoms,
            bond_feat_dim=bond_feat_dim,
            num_targets=self.num_targets,
            **self.config["model"],
        ).to(self.device)

        if distutils.is_master() and not self.silent:
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
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device], output_device=self.device
            )

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        map_location = torch.device("cpu") if self.cpu else self.device
        logging.info(f"Loading checkpoint from: {checkpoint_path} onto {map_location}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)

        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        first_key = next(iter(checkpoint["state_dict"]))
        if not distutils.initialized() and first_key.split(".")[1] == "module":
            # No need for OrderedDict since dictionaries are technically ordered
            # since Python 3.6 and officially ordered since Python 3.7
            new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(new_dict)
        elif distutils.initialized() and first_key.split(".")[1] != "module":
            new_dict = {f"module.{k}": v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(checkpoint["normalizers"][key])
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self):
        self.loss_fn = {}
        self.loss_fn["energy"] = self.config["optim"].get("loss_energy", "mae")
        self.loss_fn["force"] = self.config["optim"].get("loss_force", "mae")
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ["l1", "mae"]:
                self.loss_fn[loss] = nn.L1Loss()
            elif loss_name == "mse":
                self.loss_fn[loss] = nn.MSELoss()
            elif loss_name == "l2mae":
                self.loss_fn[loss] = L2MAELoss()
            else:
                raise NotImplementedError(f"Unknown loss function name: {loss_name}")
            if distutils.initialized():
                self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
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

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if not self.is_debug and distutils.is_master():
            if training_state:
                save_checkpoint(
                    {
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
                    },
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
        distutils.synchronize()

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

    @torch.no_grad()
    def validate(
        self, split="val", disable_tqdm=False, name_split=None, debug_batches=-1
    ):
        if distutils.is_master() and not self.silent:
            if not name_split:
                logging.info(f"Evaluating on {split}.")
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
        rank = distutils.get_rank()

        loader = self.val_loader if split[:3] in {"val", "eva"} else self.test_loader
        val_time = time.time()

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):

            if debug_batches > 0 and i == debug_batches:
                break

            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                preds = self.model_forward(batch)
            loss = self.compute_loss(preds, batch)
            if preds.get("pooling_loss") is not None:
                loss += preds["pooling_loss"]

            # Compute metrics.
            metrics = self.compute_metrics(preds, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        val_time = time.time() - val_time

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
        log_dict.update({"epoch": self.epoch})
        log_dict.update({"val_time": val_time})
        if distutils.is_master() and not self.silent:
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            if split == "eval":
                log_dict = {f"{name_split}-{k}": v for k, v in log_dict.items()}
                self.logger.log(
                    log_dict,
                    split=split,
                )
            else:
                self.logger.log(
                    log_dict,
                    step=self.step,
                    split=split,
                )
        if self.ema:
            self.ema.restore()

        return metrics

    @abstractmethod
    def model_forward(self, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def compute_loss(self, out, batch_list):
        """Derived classes should implement this function."""

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
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
            if self.logger is not None:
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
            f"{self.task_name}_{results_file}_{distutils.get_rank()}.npz",
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
                self.config["results_dir"],
                f"{self.task_name}_{results_file}.npz",
            )

            for i in range(distutils.get_world_size()):
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

    def eval_all_val_splits(
        self, final=True, disable_tqdm=True, debug_batches=-1, epoch=-1
    ):
        """Evaluate model on all four validation splits"""

        if final:
            # Load current best checkpoint
            checkpoint_path = os.path.join(
                self.config["checkpoint_dir"], "best_checkpoint.pt"
            )
            self.load_checkpoint(checkpoint_path=checkpoint_path)
            logging.info(
                "Checking models are identical:"
                + str(list(self.model.parameters())[0].data.view(-1)[:20]),
            )

        # Compute performance metrics on all four validation splits
        cumulated_time = 0
        cumulated_mae = 0
        metrics_dict = {}

        if self.task_name in OCP_TASKS:
            val_sets = ["val_ood_ads", "val_ood_cat", "val_ood_both", "val_id"]
        elif self.task_name == "qm9":
            val_sets = ["val"]
        elif self.task_name == "qm7x":
            val_sets = ["val_id", "val_ood"]
        else:
            raise ValueError(f"Unknown task {self.task_name}")

        if not self.silent:
            logging.info(f"Evaluating on {len(val_sets)} val splits.")

        for i, s in enumerate(val_sets):

            # Update the val. dataset we look at
            if self.task_name in OCP_TASKS:
                base = Path(f"/network/projects/ocp/oc20/{self.task_name}/all/{s}/")
                src = base / "data.lmdb"
                if not src.exists():
                    src = base
                self.config["val_dataset"] = {"src": str(src)}
            elif self.task_name == "qm7x":
                self.config["val_dataset"] = {**self.config["val_dataset"], "split": s}

            # Load val dataset
            if self.config.get("val_dataset", None):
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(
                    self.config["val_dataset"],
                    transform=get_transforms(self.config),
                )
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            # Call validate function
            start_time = time.time()
            self.metrics = self.validate(
                split="eval",
                disable_tqdm=disable_tqdm,
                name_split=s,
                debug_batches=debug_batches,
            )
            metrics_dict[s] = self.metrics
            cumulated_mae += self.metrics["energy_mae"]["metric"]
            cumulated_time += time.time() - start_time

        # Log time
        if final and self.config["logger"] == "wandb" and distutils.is_master():
            overall_mae = cumulated_mae / 4
            sid = os.getenv("SLURM_JOB_ID")
            self.logger.log({"Eval time": cumulated_time})
            self.logger.log({"Overall MAE": overall_mae})
            if self.logger.ntfy:
                self.logger.ntfy(
                    message=f"{sid} - Overall MAE: {overall_mae}", click=self.logger.url
                )

        # Print results
        if not self.silent:
            if final:
                print("----- FINAL RESULTS -----")
            elif epoch >= 0:
                print(f"----- RESULTS AT EPOCH {epoch} -----")
            else:
                print("----- RESULTS -----")
            print("Total time taken: ", time.time() - start_time)
            print(self.metrics.keys())
        for k, v in metrics_dict.items():
            store = []
            for _, val in v.items():
                store.append(round(val["metric"], 4))
            if not self.silent:
                print(k, store)

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
                self.config["frame_averaging"], self.config["fa_frames"]
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
                self.config["frame_averaging"], self.config["fa_frames"]
            )
            for g in g_list:
                g = fa_transform(g)
            batch_reflected = Batch.from_data_list(g_list)
            if hasattr(batch, "neighbors"):
                batch_reflected.neighbors = batch.neighbors

        return {"batch_list": [batch_reflected], "rot": rot}
