"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm

from ocpmodels.common import dist_utils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.timer import Times
from ocpmodels.common.utils import OCP_AND_DEUP_TASKS, check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer

is_test_env = os.environ.get("ocp_test_env", False)


@registry.register_trainer("single")
class SingleTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_. # noqa: E501
    """

    @property
    def now(self):
        return str(datetime.datetime.now()).split(".")[0]

    def load_task(self):
        self.num_targets = 1

        # start imports from
        # force_trainer:

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
        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config["model"].get("regress_forces"):
            if self.normalizer.get("normalize_labels", False):
                if "grad_target_mean" in self.normalizer:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.normalizer["grad_target_mean"],
                        std=self.normalizer["grad_target_std"],
                        device=self.device,
                    )
                else:
                    if not self.silent:
                        print(
                            "Warning: grad_target_mean not found in normalizer but",
                            "regress_forces and normalize_labels are true.",
                        )
                    self.normalizers["grad_target"] = Normalizer(
                        tensor=self.datasets[self.train_dataset_name].data.y[
                            self.datasets[self.train_dataset_name].__indices__
                        ],
                        device=self.device,
                    )
                    self.normalizers["grad_target"].mean.fill_(0)

    @torch.no_grad()
    def predict(self, loader, per_image=True, results_file=None, disable_tqdm=False):
        if dist_utils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = dist_utils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        if self.normalizers is not None and "grad_target" in self.normalizers:
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": []}
        if self.task_name == "s2ef":
            predictions["forces"] = []
            predictions["chunk_idx"] = []

        for batch_list in tqdm(
            loader,
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                preds = self.model_forward(batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                hofs = None
                if self.task_name == "qm7x":
                    hofs = torch.cat(
                        [batch.hofs.to(self.device) for batch in batch_list], dim=0
                    )
                preds["energy"] = self.normalizers["target"].denorm(
                    preds["energy"], hofs=hofs
                )
            if self.normalizers is not None and "grad_target" in self.normalizers:
                self.normalizers["grad_target"].to(self.device)

            if per_image:
                system_ids = (
                    [str(i) for i in batch_list[0].sid.tolist()]
                    if self.task_name == "s2ef"
                    else [
                        str(i) + "_" + str(j)
                        for i, j in zip(
                            batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                        )
                    ]
                )
                predictions["id"].extend(system_ids)
                predictions["energy"].extend(preds["energy"].to(torch.float16).tolist())

                if self.task_name == "s2ef":
                    batch_natoms = torch.cat([batch.natoms for batch in batch_list])
                    batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                    forces = preds["forces"].cpu().detach().to(torch.float16)
                    per_image_forces = torch.split(forces, batch_natoms.tolist())
                    per_image_forces = [force.numpy() for force in per_image_forces]
                    # evalAI only requires forces on free atoms
                    if results_file is not None:
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_forces = [
                            force[(fixed == 0).tolist()]
                            for force, fixed in zip(per_image_forces, _per_image_fixed)
                        ]
                        _chunk_idx = np.array(
                            [
                                free_force.shape[0]
                                for free_force in _per_image_free_forces
                            ]
                        )
                        per_image_forces = _per_image_free_forces
                        predictions["chunk_idx"].extend(_chunk_idx)
                    predictions["forces"].extend(per_image_forces)
            else:
                predictions["energy"] = preds["energy"].detach()
                if self.task_name == "s2ef":
                    predictions["forces"] = preds["forces"].detach()
                return predictions

        self.save_results(predictions, results_file, keys=["energy"])

        if self.ema:
            self.ema.restore()

        return predictions

    def train(
        self, disable_eval_tqdm=True, debug_batches=-1, save_best_ckpt_only=False
    ):
        if not torch.is_grad_enabled():
            print("\nWarning: torch grad is disabled. Enabling.\n")
            torch.set_grad_enabled(True)
        n_train = min(
            len(self.loaders[self.train_dataset_name]),
            self.config["optim"]["max_steps"],
        )
        epoch_int = 0
        eval_every = self.config["optim"].get("eval_every", n_train) or n_train
        if eval_every < 1:
            eval_every = int(n_train * eval_every)
        if self.config["print_every"] < 0:
            self.config["print_every"] = n_train
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.task_name]
        )
        if "energy_force_within_threshold" in primary_metric:
            self.best_val_metric = -np.inf
        else:
            self.best_val_metric = np.inf

        current_val_metric = None
        first_eval = True
        log_train_every = self.config["log_train_every"]
        if log_train_every < 0:
            log_train_every = n_train

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        if (
            "continue_from_dir" in self.config
            and self.config["continue_from_dir"] is not None
            and self.config["adsorbates"] not in {None, "all"}
        ):
            self.step = 0
        start_epoch = self.step // n_train
        max_epochs = self.config["optim"]["max_epochs"]
        timer = Times()
        epoch_times = []
        model_run_time = 0

        if not self.silent:
            print(f"\n--- 🔄 Beginning of Training @ {self.now} ---\n")
            print(f"Staring from epoch {start_epoch} and step {self.step}")
            print(f"Will train for {max_epochs} epochs of {n_train} steps each")
            print(f"Logging  train metrics every {log_train_every} steps")
            print(f"Printing train metrics every {self.config['print_every']} steps")
            print(f"Evaluating every {eval_every} steps\n")

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            if self.config["grad_fine_tune"]:
                if epoch_int < self.config["optim"].get("epoch_fine_tune", 1):
                    self.config["model"]["regress_forces"] = "direct"
                elif self.config["model"].get("exact_ec_pred", False):
                    self.config["model"]["regress_forces"] = "from_energy"
                    # self.config["optim"]["force_coefficient"] = 0
                else:
                    self.config["model"][
                        "regress_forces"
                    ] = "direct_with_gradient_target"
                    self.config["optim"]["force_coefficient"] = 0
                    # self.config["optim"]["energy_coefficient"] = 0
                    # print('Fine tuning gradients: change energy/force coefficients')

            start_time = time.time()
            if not self.silent:
                print()
                logging.info(f"Epoch: {epoch_int}")

            self.samplers[self.train_dataset_name].set_epoch(epoch_int)
            skip_steps = self.step % n_train
            train_loader_iter = iter(self.loaders[self.train_dataset_name])
            self.model.train()
            i_for_epoch = 0

            for i in range(skip_steps, n_train):
                if self.sigterm:
                    return "SIGTERM"
                i_for_epoch += 1
                # print(self.now, "i_for_epoch: ", i_for_epoch, flush=True)
                self.epoch = epoch_int + (i + 1) / n_train
                self.step = epoch_int * n_train + i + 1

                # Get a batch.
                with timer.next("get_batch"):
                    batch = next(train_loader_iter)

                # Forward, loss, backward.
                if epoch_int == 1:
                    s = time.time()

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    with timer.next("train_forward", ignore=epoch_int > 0):
                        preds = self.model_forward(batch)
                    loss = self.compute_loss(preds, batch)

                if epoch_int == 1:
                    model_run_time += time.time() - s

                loss = {
                    k: self.scaler.scale(v) if self.scaler else v
                    for k, v in loss.items()
                }

                if torch.isnan(loss["total_loss"]):
                    print("\n\n >>> 🛑 Loss is NaN. Stopping training.\n\n")
                    self.logger.add_tags(["nan_loss"])
                    return "loss_is_nan"

                try:
                    with timer.next("train_backward", ignore=epoch_int > 0):
                        self._backward(loss)
                except RuntimeError:
                    print("\nBackward loss issue")
                    print(loss)
                    print(
                        "Requires grad:",
                        {k: v.requires_grad for k, v in loss.items()},
                    )
                    print()
                # Compute metrics.
                self.metrics = self.compute_metrics(
                    preds,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                scale = self.scaler.get_scale() if self.scaler else 1.0

                if i_for_epoch % log_train_every == 0:
                    for k, v in loss.items():
                        self.metrics = self.evaluator.update(
                            k, v.item() / scale, self.metrics
                        )

                    # Log metrics.
                    gbm, gbs = timer.prepare_for_logging()
                    self.metrics["get_batch_time_mean"] = {"metric": gbm["get_batch"]}
                    self.metrics["get_batch_time_std"] = {"metric": gbs["get_batch"]}
                    timer.reset("get_batch")
                    # logging.info(f"Step: {self.step}")
                    self.log_train_metrics()

                is_final_epoch = epoch_int == self.config["optim"]["max_epochs"] - 1
                is_final_batch = (i == n_train - 1) or (
                    debug_batches > 0 and i_for_epoch == debug_batches
                )

                if is_test_env:
                    if is_final_batch:
                        break
                    continue

                should_validate = (self.step % eval_every == 0) or (
                    is_final_epoch and is_final_batch
                )
                primary_metric = self.evaluator.task_primary_metric[self.task_name]

                # Evaluate on val set after every `eval_every` iterations.
                if should_validate:
                    if not save_best_ckpt_only:
                        self.save(
                            checkpoint_file=f"checkpoint-{str(self.step).zfill(7)}.pt",
                            training_state=True,
                        )

                    val_metrics = self.validate(
                        split=self.config["dataset"]["default_val"],
                        disable_tqdm=disable_eval_tqdm,
                        debug_batches=debug_batches,
                        is_first=first_eval,
                    )

                    first_eval = False
                    if val_metrics == "SIGTERM":
                        return "SIGTERM"

                    current_val_metric = val_metrics[primary_metric]["metric"]

                    if (
                        primary_metric in {"energy_mae", "forces_mae", "energy_mse"}
                        and current_val_metric < self.best_val_metric
                    ) or (
                        "energy_force_within_threshold" in primary_metric
                        and current_val_metric > self.best_val_metric
                    ):
                        # if current_val_metric < self.best_val_metric:
                        self.best_val_metric = current_val_metric
                        self.save(
                            metrics=val_metrics,
                            checkpoint_file="best_checkpoint.pt",
                            training_state=False,
                        )
                    if (
                        self.early_stopper.should_stop(
                            current_val_metric, self.scheduler.get_lr(), self.epoch
                        )
                        or self.early_stopping_file.exists()
                    ):
                        if self.early_stopping_file.exists():
                            print("\n\n >>> 🛑 Early stopping file found.\n\n")
                            now = self.now.replace(" ", "_").replace(":", "-")
                            self.early_stopping_file.rename(
                                self.early_stopping_file.parent
                                / f"{self.early_stopping_file.stem}_{now}.stopped"
                            )
                        else:
                            print(f"\n\n >>> 🛑 {self.early_stopper.reason}\n\n")

                        if self.logger:
                            self.logger.add_tags(["E-S"])
                        return self.end_of_training(
                            epoch_int, debug_batches, model_run_time, epoch_times
                        )

                    self.model.train()

                self.scheduler_step(eval_every, current_val_metric)

                if is_final_batch:
                    break

                # End of batch.

            # End of epoch.
            epoch_times.append(time.time() - start_time)
            self.metrics["epoch_time"] = {"metric": epoch_times[-1]}
            if epoch_int == 0:
                tm, ts = timer.prepare_for_logging(
                    map_funcs={
                        "train_backward": lambda x: x
                        / self.config["optim"]["batch_size"],
                        "train_forward": lambda x: x
                        / self.config["optim"]["batch_size"],
                    }
                )
                self.metrics["train_backward_mean"] = {"metric": tm["train_backward"]}
                self.metrics["train_backward_std"] = {"metric": ts["train_backward"]}
                self.metrics["train_forward_mean"] = {"metric": tm["train_forward"]}
                self.metrics["train_forward_std"] = {"metric": ts["train_forward"]}
            self.log_train_metrics(end_of_epoch=True)
            torch.cuda.empty_cache()

        # End of training.
        if not is_test_env:
            if self.config["model"].get("exact_ec_pred", False):
                self.config["model"]["regress_forces"] = "from_energy"
            return self.end_of_training(
                epoch_int, debug_batches, model_run_time, epoch_times
            )

    def end_of_training(
        self,
        epoch_int,
        debug_batches,
        model_run_time,
        epoch_times,
        from_ckpt=None,
        disable_tqdm=True,
    ):
        eas = self.eval_all_splits(
            True,
            epoch=epoch_int,
            debug_batches=debug_batches,
            from_ckpt=from_ckpt,
            disable_tqdm=disable_tqdm,
        )
        if eas == "SIGTERM":
            return "SIGTERM"

        if "test" in self.loaders:
            # TODO: update predict function
            # self.predict(self.loaders["test"], results_file="predictions")
            pass

        # Time model
        if self.logger is not None:
            log_epoch_times = self.config["optim"]["max_epochs"] > 0
            start_time = time.time()

            # deterministic batch because shuffle=False for validation
            batch = next(iter(self.loaders[self.config["dataset"]["default_val"]]))
            self.model_forward(batch)
            self.logger.log({"Batch time": time.time() - start_time})
            self.logger.log(
                {
                    "Model run time": model_run_time
                    / len(self.loaders[self.train_dataset_name])
                }
            )
            if log_epoch_times:
                self.logger.log({"Epoch time": np.mean(epoch_times)})

        # Check respect of symmetries
        if self.test_ri and not is_test_env:
            symmetry = self.test_model_symmetries(debug_batches=debug_batches)
            if symmetry == "SIGTERM":
                return "SIGTERM"
            if self.logger:
                self.logger.log(symmetry)
            if not self.silent:
                print(symmetry)

        # Close datasets
        if debug_batches < 0:
            for ds in self.datasets.values():
                try:
                    ds.close_db()
                except:
                    assert self.config["lowest_energy_only"] == True
                    self.real_dataset.close_db()

    def model_forward(self, batch_list, mode="train", q=None):
        """Perform a forward pass of the model when frame averaging is applied.
        Returns:
            (dict): model predictions tensor for "energy" and "forces".
        """
        # Distinguish frame averaging from base case.
        if self.config["frame_averaging"] and self.config["frame_averaging"] != "DA":
            original_pos = batch_list[0].pos
            if self.task_name in OCP_AND_DEUP_TASKS:
                original_cell = batch_list[0].cell
            e_all, f_all, gt_all = [], [], []

            # Compute model prediction for each frame
            for i in range(len(batch_list[0].fa_pos)):
                batch_list[0].pos = batch_list[0].fa_pos[i]
                if self.task_name in OCP_AND_DEUP_TASKS:
                    batch_list[0].cell = batch_list[0].fa_cell[i]

                # forward pass
                preds = self.model(
                    deepcopy(batch_list),
                    mode=mode,
                    regress_forces=self.config["model"]["regress_forces"],
                    q=q,
                )
                e_all.append(preds["energy"])

                fa_rot = None

                if preds.get("forces") is not None:
                    # Transform forces to guarantee equivariance of FA method
                    fa_rot = torch.repeat_interleave(
                        batch_list[0].fa_rot[i], batch_list[0].natoms, dim=0
                    )
                    g_forces = (
                        preds["forces"]
                        .view(-1, 1, 3)
                        .bmm(fa_rot.transpose(1, 2).to(preds["forces"].device))
                        .view(-1, 3)
                    )
                    f_all.append(g_forces)
                if preds.get("forces_grad_target") is not None:
                    # Transform gradients to stay consistent with FA
                    if fa_rot is None:
                        fa_rot = torch.repeat_interleave(
                            batch_list[0].fa_rot[i], batch_list[0].natoms, dim=0
                        )
                    g_grad_target = (
                        preds["forces_grad_target"]
                        .view(-1, 1, 3)
                        .bmm(
                            fa_rot.transpose(1, 2).to(
                                preds["forces_grad_target"].device
                            )
                        )
                        .view(-1, 3)
                    )
                    gt_all.append(g_grad_target)

            batch_list[0].pos = original_pos
            if self.task_name in OCP_AND_DEUP_TASKS:
                batch_list[0].cell = original_cell

            # Average predictions over frames
            preds["energy"] = sum(e_all) / len(e_all)
            if len(f_all) > 0 and all(y is not None for y in f_all):
                preds["forces"] = sum(f_all) / len(f_all)
            if len(gt_all) > 0 and all(y is not None for y in gt_all):
                preds["forces_grad_target"] = sum(gt_all) / len(gt_all)
        else:
            preds = self.model(batch_list)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"].view(-1)

        return preds

    def compute_loss(self, preds, batch_list):
        loss = {"total_loss": []}

        # Energy loss
        energy_target = torch.cat(
            [
                (
                    batch.y_relaxed.to(self.device)
                    if self.task_name == "is2re"
                    else (
                        batch.deup_loss.to(self.device)
                        if self.task_name == "deup_is2re"
                        else batch.y.to(self.device)
                    )
                )
                for batch in batch_list
            ],
            dim=0,
        )

        if self.normalizer.get("normalize_labels", False):
            hofs = None
            if self.task_name == "qm7x":
                hofs = torch.cat(
                    [batch.hofs.to(self.device) for batch in batch_list], dim=0
                )
            target_normed = self.normalizers["target"].norm(energy_target, hofs=hofs)
        else:
            target_normed = energy_target
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss["energy_loss"] = self.loss_fn["energy"](preds["energy"], target_normed)
        loss["total_loss"].append(energy_mult * loss["energy_loss"])

        # Force loss.
        if self.task_name in {"is2rs", "s2ef"} or self.config["model"].get(
            "regress_forces"
        ):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if (
                self.normalizer.get("normalize_labels", False)
                and "grad_target" in self.normalizers
            ):
                force_target = self.normalizers["grad_target"].norm(force_target)

            tag_specific_weights = self.config["task"].get("tag_specific_weights", [])
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [batch.tags.float().to(self.device) for batch in batch_list],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(preds["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                dist_utils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    dist_utils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                mask = torch.ones_like(force_target).bool().to(self.device)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0

                loss["force_loss"] = self.loss_fn["force"](
                    preds["forces"][mask], force_target[mask]
                )
                loss["total_loss"].append(force_mult * loss["force_loss"])
                if "forces_grad_target" in preds:
                    grad_target = preds["forces_grad_target"]
                    if self.config["model"].get("cosine_sim", False):
                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        loss["energy_grad_loss"] = -torch.mean(
                            cos(preds["forces"][mask], grad_target[mask])
                        )
                    else:
                        loss["energy_grad_loss"] = self.loss_fn["force"](
                            preds["forces"][mask], grad_target[mask]
                        )
                    if (
                        self.config["model"].get("regress_forces")
                        == "direct_with_gradient_target"
                    ):
                        energy_grad_mult = self.config["optim"].get(
                            "energy_grad_coefficient", 10
                        )
                        loss["total_loss"].append(
                            energy_grad_mult * loss["energy_grad_loss"]
                        )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss["total_loss"]:
            assert hasattr(lc, "grad_fn")

        loss["total_loss"] = sum(loss["total_loss"])
        return loss

    def compute_metrics(
        self, preds: Dict, batch_list: List[Data], evaluator: Evaluator, metrics={}
    ):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [
                    (
                        batch.y_relaxed.to(self.device)
                        if self.task_name == "is2re"
                        else (
                            batch.deup_loss.to(self.device)
                            if self.task_name == "deup_is2re"
                            else batch.y.to(self.device)
                        )
                    )
                    for batch in batch_list
                ],
                dim=0,
            ),
            "natoms": natoms,
        }

        if self.config["model"].get("regress_forces", False):
            target["forces"] = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            preds["natoms"] = natoms

            if "forces_grad_target" in preds:
                target["forces_grad_target"] = preds["forces_grad_target"]

            if (
                self.config["task"].get("eval_on_free_atoms", False)
                and self.task_name in OCP_AND_DEUP_TASKS
            ):
                fixed = torch.cat([batch.fixed.to(self.device) for batch in batch_list])
                mask = fixed == 0
                preds["forces"] = preds["forces"][mask]
                target["forces"] = target["forces"][mask]
                if "forces_grad_target" in target:
                    target["forces_grad_target"] = target["forces_grad_target"][mask]

                s_idx = 0
                natoms_free = []
                for natoms in target["natoms"]:
                    natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                    s_idx += natoms
                target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
                preds["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            if (
                self.normalizer.get("normalize_labels")
                and "grad_target" in self.normalizers
            ):
                if not self.config.get("no_metrics_denorm"):
                    preds["forces"] = self.normalizers["grad_target"].denorm(
                        preds["forces"]
                    )
                else:
                    target["forces"] = self.normalizers["grad_target"].norm(
                        target["forces"]
                    )

        if self.normalizer.get("normalize_labels") and "target" in self.normalizers:
            hofs = None
            if self.task_name == "qm7x":
                hofs = torch.cat(
                    [batch.hofs.to(self.device) for batch in batch_list], dim=0
                )
            if not self.config.get("no_metrics_denorm"):
                preds["energy"] = self.normalizers["target"].denorm(
                    preds["energy"], hofs=hofs
                )
            else:
                target["energy"] = self.normalizers["target"].norm(
                    target["energy"], hofs=hofs
                )

        metrics = evaluator.eval(preds, target, prev_metrics=metrics)

        return metrics

    def log_train_metrics(self, end_of_epoch=False):
        log_dict = {k: v["metric"] for k, v in self.metrics.items()}
        log_dict.update(
            {
                "lr": self.scheduler.get_lr(),
                "epoch": self.epoch,
                "step": self.step,
            }
        )
        if (
            self.step % self.config["print_every"] == 0
            and dist_utils.is_master()
            and not self.is_hpo
        ) or (dist_utils.is_master() and end_of_epoch):
            if not self.silent:
                log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                print(
                    f"\nTrain metrics at step {self.step}:\n  > "
                    + "\n  > ".join(log_str)
                )
            self.metrics = {}

        if self.logger is not None:  # and not end_of_epoch:
            self.logger.log(
                log_dict,
                step=self.step,
                split="train",
            )

    @torch.no_grad()
    def test_model_symmetries(self, debug_batches=-1):
        """Test rotation and reflection invariance & equivariance
        of GNNs

        Returns:
            (tensors): metrics to measure RI difference in
            energy/force pred. or pos. between G and rotated G
        """
        if not self.silent:
            logging.info("Testing model symmetries")

        self.model.eval()

        energy_diff = torch.zeros(1, device=self.device)
        energy_diff_z = torch.zeros(1, device=self.device)
        energy_diff_z_percentage = torch.zeros(1, device=self.device)
        energy_diff_refl = torch.zeros(1, device=self.device)
        pos_diff_total = torch.zeros(1, device=self.device)
        forces_diff = torch.zeros(1, device=self.device)
        forces_diff_z = torch.zeros(1, device=self.device)
        forces_diff_z_graph = torch.zeros(1, device=self.device)
        forces_diff_refl = torch.zeros(1, device=self.device)
        n_batches = 0
        n_atoms = 0

        for i, batch in enumerate(self.loaders[self.config["dataset"]["default_val"]]):
            if self.sigterm:
                return "SIGTERM"
            if debug_batches > 0 and i == debug_batches:
                break

            n_batches += len(batch[0].natoms)
            n_atoms += batch[0].natoms.sum()

            # Compute model prediction
            preds1 = self.model_forward(deepcopy(batch), mode="inference")

            # Compute prediction on rotated graph
            rotated = self.rotate_graph(batch, rotation="z")
            preds2 = self.model_forward(
                deepcopy(rotated["batch_list"]), mode="inference"
            )

            # Difference in predictions, for energy and forces
            energy_diff_z += torch.abs(preds1["energy"] - preds2["energy"]).sum()

            if self.task_name == "s2ef":
                energy_diff_z_percentage += (
                    torch.abs(preds1["energy"] - preds2["energy"])
                    / torch.abs(batch[0].y).to(preds1["energy"].device)
                ).sum()
                forces_diff_z += torch.abs(
                    preds1["forces"] @ rotated["rot"].to(preds1["forces"].device)
                    - preds2["forces"]
                ).sum()
                assert torch.allclose(
                    torch.abs(
                        batch[0].force @ rotated["rot"].to(batch[0].force.device)
                        - rotated["batch_list"][0].force
                    ).sum(),
                    torch.tensor([0.0]),
                    atol=1e-05,
                )
            elif self.task_name == "is2re":
                energy_diff_z_percentage += (
                    torch.abs(preds1["energy"] - preds2["energy"])
                    / torch.abs(batch[0].y_relaxed).to(preds1["energy"].device)
                ).sum()
            else:
                energy_diff_z_percentage += (
                    torch.abs(preds1["energy"] - preds2["energy"])
                    / torch.abs(batch[0].y).to(preds1["energy"].device)
                ).sum()

            # Diff in positions
            pos_diff = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff = 0
                # Compute total difference across frames
                for pos1, pos2 in zip(batch[0].fa_pos, rotated["batch_list"][0].fa_pos):
                    pos_diff += pos1 - pos2
                # Manhattan distance of pos matrix wrt 0 matrix.
                pos_diff_total += torch.abs(pos_diff).sum()

            # Reflect graph and compute diff in prediction
            reflected = self.reflect_graph(batch)
            preds3 = self.model_forward(reflected["batch_list"], mode="inference")
            energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()
            if self.task_name == "s2ef":
                forces_diff_refl += torch.abs(
                    preds1["forces"] @ reflected["rot"].to(preds1["forces"].device)
                    - preds3["forces"]
                ).sum()
                # assert torch.allclose(
                #     torch.abs(
                #         batch[0].force @ reflected["rot"].to(batch[0].force.device)
                #         - reflected["batch_list"][0].force #.to(batch[0].force.device)
                #     ).sum(),
                #     torch.tensor([0.0]),   # .to(batch[0].force.device)
                #     atol=1e-05,
                # )

            # 3D Rotation and compute diff in prediction
            rotated = self.rotate_graph(batch)
            preds4 = self.model_forward(rotated["batch_list"], mode="inference")
            energy_diff += torch.abs(preds1["energy"] - preds4["energy"]).sum()
            if self.task_name == "s2ef":
                forces_diff += torch.abs(preds1["forces"] - preds4["forces"]).sum()

        # Aggregate the results
        energy_diff_z = energy_diff_z / n_batches
        energy_diff_z_percentage = energy_diff_z_percentage / n_batches
        energy_diff = energy_diff / n_batches
        energy_diff_refl = energy_diff_refl / n_batches
        pos_diff_total = pos_diff_total / n_batches

        symmetry = {
            "2D_E_ri": float(energy_diff_z),
            "2D_E_ri_percentage": float(energy_diff_z_percentage),
            "3D_E_ri": float(energy_diff),
            "2D_pos_ri": float(pos_diff_total),
            "2D_E_refl_i": float(energy_diff_refl),
        }

        # Test equivariance of forces
        if self.task_name == "s2ef":
            forces_diff_z = forces_diff_z / n_atoms
            forces_diff_z_graph = forces_diff_z / n_batches
            forces_diff = forces_diff / n_atoms
            forces_diff_refl = forces_diff_refl / n_atoms
            symmetry.update(
                {
                    "2D_F_ri_graph": float(forces_diff_z_graph),
                    "2D_F_ri": float(forces_diff_z),
                    "3D_F_ri": float(forces_diff),
                    "2D_F_refl_i": float(forces_diff_refl),
                }
            )

        if not self.silent:
            logging.info("Symmetry results:")
            print("".join([f"\n  > {k:12}: {v:.5f}" for k, v in symmetry.items()]))

        return symmetry

    def run_relaxations(self, split="val"):
        assert self.task_name == "s2ef"
        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs = Evaluator(
            task="is2rs",
            model_regresses_forces=self.config["model"].get("regress_forces", ""),
        )
        evaluator_is2re = Evaluator(
            task="is2re",
            model_regresses_forces=self.config["model"].get("regress_forces", ""),
        )

        metrics_is2rs = {}
        metrics_is2re = {}

        if hasattr(self.relax_dataset[0], "pos_relaxed") and hasattr(
            self.relax_dataset[0], "y_relaxed"
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = dist_utils.get_rank()
            pos_filename = os.path.join(
                self.config["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            dist_utils.synchronize()
            if dist_utils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(dist_utils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(rank_results["chunk_idx"])
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": dist_utils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": dist_utils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {f"{task}_{k}": metrics[k]["metric"] for k in metrics}
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if dist_utils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()
