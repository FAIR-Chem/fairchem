"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("energy")
class EnergyTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_.


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_vis (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        logger_project (str, optional): Logger project to save results in (wandb only).
            (default: :obj:`None`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
    """

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
        logger="tensorboard",
        logger_project=None,
        local_rank=0,
        amp=False,
        cpu=False,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            run_dir=run_dir,
            is_debug=is_debug,
            is_vis=is_vis,
            print_every=print_every,
            seed=seed,
            logger=logger,
            logger_project=logger_project,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
        )

    def load_task(self):
        assert (
            self.config["task"]["dataset"] == "single_point_lmdb"
        ), "EnergyTrainer requires single_point_lmdb dataset"

        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        self.parallel_collater = ParallelCollater(
            1 if not self.cpu else 0,
            self.config["model_attributes"].get("otf_graph", False),
        )

        self.train_dataset = registry.get_dataset_class(
            self.config["task"]["dataset"]
        )(self.config["dataset"])

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            shuffle=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["optim"]["batch_size"],
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            sampler=self.train_sampler,
        )

        self.val_loader = self.test_loader = None
        self.val_sampler = None

        if "val_dataset" in self.config:
            self.val_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["val_dataset"])
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=distutils.get_world_size(),
                rank=distutils.get_rank(),
                shuffle=False,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                self.config["optim"].get("eval_batch_size", 64),
                collate_fn=self.parallel_collater,
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=True,
                sampler=self.val_sampler,
            )
        if "test_dataset" in self.config:
            self.test_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["test_dataset"])
            self.test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=distutils.get_world_size(),
                rank=distutils.get_rank(),
                shuffle=False,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                self.config["optim"].get("eval_batch_size", 64),
                collate_fn=self.parallel_collater,
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=True,
                sampler=self.test_sampler,
            )

        self.num_targets = 1

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config["dataset"].get("normalize_labels", False):
            if "target_mean" in self.config["dataset"]:
                self.normalizers["target"] = Normalizer(
                    mean=self.config["dataset"]["target_mean"],
                    std=self.config["dataset"]["target_std"],
                    device=self.device,
                )
            else:
                raise NotImplementedError

    @torch.no_grad()
    def predict(self, loader, results_file=None, disable_tqdm=False):
        if distutils.is_master() and not disable_tqdm:
            print("### Predicting on test.")
        assert isinstance(loader, torch.utils.data.dataloader.DataLoader)
        rank = distutils.get_rank()

        self.model.eval()
        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        predictions = {"id": [], "energy": []}

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
            predictions["id"].extend([str(i) for i in batch[0].sid.tolist()])
            predictions["energy"].extend(out["energy"].tolist())

        self.save_results(predictions, results_file, keys=["energy"])
        return predictions

    def train(self):
        self.best_val_mae = 1e9

        start_epoch = self.start_step // len(self.train_loader)
        for epoch in range(start_epoch, self.config["optim"]["max_epochs"]):
            self.train_sampler.set_epoch(epoch)
            self.model.train()

            skip_steps = 0
            if epoch == start_epoch and start_epoch > 0:
                skip_steps = start_epoch % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                batch = next(train_loader_iter)
                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Print metrics, make plots.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {"epoch": epoch + (i + 1) / len(self.train_loader)}
                )
                if (
                    i % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [
                        "{}: {:.4f}".format(k, v) for k, v in log_dict.items()
                    ]
                    print(", ".join(log_str))

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                if self.update_lr_on_step:
                    self.scheduler.step()

            if not self.update_lr_on_step:
                self.scheduler.step()

            torch.cuda.empty_cache()

            if self.val_loader is not None:
                val_metrics = self.validate(split="val", epoch=epoch)
                if (
                    val_metrics[self.evaluator.task_primary_metric[self.name]][
                        "metric"
                    ]
                    < self.best_val_mae
                ):
                    self.best_val_mae = val_metrics[
                        self.evaluator.task_primary_metric[self.name]
                    ]["metric"]
                    current_step = (epoch + 1) * len(self.train_loader)
                    self.save(epoch + 1, current_step, val_metrics)
                    if self.test_loader is not None:
                        self.predict(
                            self.test_loader,
                            results_file="predictions",
                            disable_tqdm=False,
                        )
            else:
                current_step = (epoch + 1) * len(self.train_loader)
                self.save(epoch + 1, current_step, self.metrics)

        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        output = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {
            "energy": output,
        }

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.config["dataset"].get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.criterion(out["energy"], target_normed)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.config["dataset"].get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        return metrics
