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
from abc import ABC
from collections import defaultdict
from typing import DefaultDict, Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import BalancedBatchSampler, OCPCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance as aii
from ocpmodels.common.typing import none_throws
from ocpmodels.common.utils import (
    get_commit_hash,
    get_loss_module,
    load_state_dict,
    save_checkpoint,
    update_config,
)
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import DDPLoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.scheduler import LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier: str,
        timestamp_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        is_debug: bool = False,
        print_every: int = 100,
        seed: Optional[int] = None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        name: str = "ocp",
        slurm={},
        noddp: bool = False,
    ) -> None:
        self.name = name
        self.is_debug = is_debug
        self.cpu = cpu
        self.epoch = 0
        self.step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        self.timestamp_id: str
        if timestamp_id is None:
            timestamp_id = self._get_timestamp(self.device, identifier)

        self.timestamp_id = none_throws(timestamp_id)

        commit_hash = get_commit_hash()

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "trainer": name,
            "model": aii(model.pop("name"), str),
            "model_attributes": model,
            "outputs": outputs,
            "optim": optimizer,
            "loss_fns": loss_fns,
            "eval_metrics": eval_metrics,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "noddp": noddp,
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        # Fill in SLURM information in config, if applicable
        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "%s_%s" % (
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])

        # Define datasets
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", None)
            self.config["val_dataset"] = dataset.get("val", None)
            self.config["test_dataset"] = dataset.get("test", None)
        else:
            self.config["dataset"] = dataset

        if not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        ### backwards compatability with OCP v<2.0
        ### TODO: better format check for older configs
        if not self.config.get("loss_fns"):
            logging.warning(
                "Detected old config, converting to new format. Consider updating to avoid potential incompatibilities."
            )
            self.config = update_config(self.config)

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))

        self.load()

    @staticmethod
    def _get_timestamp(device: torch.device, suffix: Optional[str]) -> str:
        now = datetime.datetime.now().timestamp()
        timestamp_tensor = torch.tensor(now).to(device)
        # create directories from master rank only
        distutils.broadcast(timestamp_tensor, 0)
        timestamp_str = datetime.datetime.fromtimestamp(
            timestamp_tensor.float().item()
        ).strftime("%Y-%m-%d-%H-%M-%S")
        if suffix:
            timestamp_str += "-" + suffix
        return timestamp_str

    def load(self) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()

    def load_seed_from_config(self) -> None:
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

    def load_logger(self) -> None:
        self.logger = None
        if not self.is_debug and distutils.is_master():
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            self.logger = registry.get_logger_class(logger_name)(self.config)

    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool
    ) -> BalancedBatchSampler:
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler) -> DataLoader:
        loader = DataLoader(
            dataset,
            collate_fn=self.ocp_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader

    def load_datasets(self) -> None:
        self.ocp_collater = OCPCollater(
            self.config["model_attributes"].get("otf_graph", False)
        )
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # load train, val, test datasets
        if self.config["dataset"].get("src", None):
            logging.info(
                f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
            )

            self.train_dataset = registry.get_dataset_class(
                self.config["dataset"].get("format", "lmdb")
            )(self.config["dataset"])
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
                if self.config["val_dataset"].get("use_train_settings", True):
                    val_config = self.config["dataset"].copy()
                    val_config.update(self.config["val_dataset"])

                self.val_dataset = registry.get_dataset_class(
                    self.config["val_dataset"].get("format", "lmdb")
                )(val_config)
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
                if self.config["test_dataset"].get("use_train_settings", True):
                    test_config = self.config["dataset"].copy()
                    test_config.update(self.config["test_dataset"])

                self.test_dataset = registry.get_dataset_class(
                    self.config["test_dataset"].get("format", "lmdb")
                )(test_config)
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

        # load relaxation dataset
        if "relax_dataset" in self.config["task"]:
            self.relax_dataset = registry.get_dataset_class("lmdb")(
                self.config["task"]["relax_dataset"]
            )
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

    def load_task(self):
        # Normalizer for the dataset.
        normalizer = (
            self.config["dataset"].get("transforms", {}).get("normalizer", {})
        )
        self.normalizers = {}
        if normalizer:
            for target in normalizer:
                self.normalizers[target] = Normalizer(
                    mean=normalizer[target].get("mean", 0),
                    std=normalizer[target].get("stdev", 1),
                )

        self.output_targets = {}
        for target_name in self.config["outputs"]:
            self.output_targets[target_name] = self.config["outputs"][
                target_name
            ]
            if "decomposition" in self.config["outputs"][target_name]:
                for subtarget in self.config["outputs"][target_name][
                    "decomposition"
                ]:
                    self.output_targets[subtarget] = (
                        self.config["outputs"][target_name]["decomposition"]
                    )[subtarget]
                    self.output_targets[subtarget]["parent"] = target_name
                    # inherent properties if not available
                    if "level" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["level"] = self.config[
                            "outputs"
                        ][target_name].get("level", "system")
                    if (
                        "train_on_free_atoms"
                        not in self.output_targets[subtarget]
                    ):
                        self.output_targets[subtarget][
                            "train_on_free_atoms"
                        ] = self.config["outputs"][target_name].get(
                            "train_on_free_atoms", True
                        )
                    if (
                        "eval_on_free_atoms"
                        not in self.output_targets[subtarget]
                    ):
                        self.output_targets[subtarget][
                            "eval_on_free_atoms"
                        ] = self.config["outputs"][target_name].get(
                            "eval_on_free_atoms", True
                        )

        # TODO: Assert that all targets, loss fn, metrics defined are consistent
        self.evaluation_metrics = self.config.get("eval_metrics", {})
        self.evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

    def load_model(self) -> None:
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
            1,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model.to(self.device)
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: Dict = {}
    ) -> None:
        if not checkpoint:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            else:
                logging.info(f"Loading checkpoint from: {checkpoint_path}")
                map_location = torch.device("cpu") if self.cpu else self.device
                checkpoint = torch.load(
                    checkpoint_path, map_location=map_location
                )

        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key in checkpoint["normalizers"]:
            ### Convert old normalizer keys to new target keys
            if key == "target":
                target_key = "energy"
            elif key == "grad_target":
                target_key = "forces"
            else:
                target_key = key

            if target_key in self.normalizers:
                self.normalizers[target_key].load_state_dict(
                    checkpoint["normalizers"][key]
                )

        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self) -> None:
        self.loss_fns = []
        for idx, loss in enumerate(self.config["loss_fns"]):
            for target in loss:
                loss_name = loss[target].get("fn", "mae")
                coefficient = loss[target].get("coefficient", 1)
                loss_reduction = loss[target].get("reduction", "mean")

                ### if torch module name provided, use that directly
                if hasattr(nn, loss_name):
                    loss_fn = getattr(nn, loss_name)()
                ### otherwise, retrieve the correct module based off old naming
                else:
                    loss_fn = get_loss_module(loss_name)

                loss_fn = DDPLoss(loss_fn, loss_name, loss_reduction)

                self.loss_fns.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

    def load_optimizer(self) -> None:
        optimizer = getattr(
            torch.optim, self.config["optim"].get("optimizer", "AdamW")
        )
        optimizer_params = self.config["optim"].get("optimizer_params", {})

        weight_decay = optimizer_params.get("weight_decay", 0)
        if "weight_decay" in self.config["optim"]:
            weight_decay = self.config["optim"]["weight_decay"]
            logging.warning(
                "Using `weight_decay` from `optim` instead of `optim.optimizer_params`."
                "Please update your config to use `optim.optimizer_params.weight_decay`."
                "`optim.weight_decay` will soon be deprecated."
            )

        if weight_decay > 0:
            self.model_params_no_wd = {}
            if hasattr(self._unwrapped_model, "no_weight_decay"):
                self.model_params_no_wd = (
                    self._unwrapped_model.no_weight_decay()
                )

            params_decay, params_no_decay, name_no_decay = [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if any(
                    name.endswith(skip_name)
                    for skip_name in self.model_params_no_wd
                ):
                    params_no_decay.append(param)
                    name_no_decay.append(name)
                else:
                    params_decay.append(param)

            if distutils.is_master():
                logging.info("Parameters without weight decay:")
                logging.info(name_no_decay)

            self.optimizer = optimizer(
                params=[
                    {"params": params_no_decay, "weight_decay": 0},
                    {"params": params_decay, "weight_decay": weight_decay},
                ],
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )

    def load_extras(self) -> None:
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = aii(
            self.config["optim"].get("clip_grad_norm", None), (int, float)
        )
        self.ema_decay = aii(self.config["optim"].get("ema_decay"), float)
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
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ) -> Optional[str]:
        if not self.is_debug and distutils.is_master():
            if training_state:
                return save_checkpoint(
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
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
                return ckpt_path
        return None

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

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
                batch.to(self.device)
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
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
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
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

    def save_results(
        self, predictions, results_file: Optional[str], keys=None
    ) -> None:
        if results_file is None:
            return
        if keys is None:
            keys = predictions.keys()

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            **{key: predictions[key] for key in keys},
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results: DefaultDict[
                str, npt.NDArray[np.float_]
            ] = defaultdict(list)
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
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            for k in keys:
                if "chunk_idx" in k:
                    gather_results[k] = np.cumsum(
                        np.array(
                            gather_results[k],
                        )[idx]
                    )[:-1]
                else:
                    if f"{k}_chunk_idx" in keys or k == "forces":
                        gather_results[k] = np.concatenate(
                            np.array(gather_results[k])[idx]
                        )
                    else:
                        gather_results[k] = np.array(gather_results[k])[idx]

            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)
