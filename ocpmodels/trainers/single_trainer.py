"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import OCP_TASKS, check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.common.timer import Times

is_test_env = os.environ.get("ocp_test_env", False)


@registry.register_trainer("single")
class SingleTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_. # noqa: E501
    """

    def load_task(self):
        if not self.silent:
            logging.info(f"Loading dataset: {self.config['task']['dataset']}")
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
                    self.normalizers["grad_target"] = Normalizer(
                        tensor=self.datasets["train"].data.y[
                            self.datasets["train"].__indices__
                        ],
                        device=self.device,
                    )
                    self.normalizers["grad_target"].mean.fill_(0)

    @torch.no_grad()
    def predict(self, loader, per_image=True, results_file=None, disable_tqdm=False):
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

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
                preds["energy"] = self.normalizers["target"].denorm(preds["energy"])
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

    def train(self, disable_eval_tqdm=True, debug_batches=-1):
        n_train = len(self.loaders["train"])
        epoch_int = 0
        eval_every = self.config["optim"].get("eval_every", n_train)
        if self.config["print_every"] < 0:
            self.config["print_every"] = n_train
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.task_name]
        )
        self.best_val_metric = np.inf
        current_val_metric = None
        first_eval = True
        log_train_every = self.config["log_train_every"]

        print(f"Logging  train metrics every {log_train_every} steps")
        print(f"Printing train metrics every {self.config['print_every']} steps")

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // n_train
        loader_times = Times()
        epoch_times = []

        if not self.silent:
            print("---Beginning of Training---")

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):

            start_time = time.time()
            if not self.silent:
                print()
                logging.info(f"Epoch: {epoch_int}")

            self.samplers["train"].set_epoch(epoch_int)
            skip_steps = self.step % n_train
            train_loader_iter = iter(self.loaders["train"])
            self.model.train()
            i_for_epoch = 0

            for i in range(skip_steps, n_train):
                if self.sigterm:
                    return "SIGTERM"
                i_for_epoch += 1
                self.epoch = epoch_int + (i + 1) / n_train
                self.step = epoch_int * n_train + i + 1

                # Get a batch.
                with loader_times.next("get_batch"):
                    batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    preds = self.model_forward(batch)
                    loss = self.compute_loss(preds, batch)
                    if preds.get("pooling_loss") is not None:
                        coeff = self.config["optim"].get("pooling_coefficient", 1)
                        loss["total_loss"] += preds["pooling_loss"] * coeff

                loss = {
                    k: self.scaler.scale(v) if self.scaler else v
                    for k, v in loss.items()
                }

                if torch.isnan(loss["total_loss"]):
                    print("\n\n >>> 🛑 Loss is NaN. Stopping training.\n\n")
                    self.logger.add_tags(["nan_loss"])
                    return True
                self._backward(loss)

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
                    gbm, gbs = loader_times.prepare_for_logging()
                    self.metrics["get_batch_time_mean"] = {"metric": gbm["get_batch"]}
                    self.metrics["get_batch_time_std"] = {"metric": gbs["get_batch"]}
                    loader_times.reset()
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
                    self.save(
                        checkpoint_file=f"checkpoint-{str(self.step).zfill(6)}.pt",
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
                    if current_val_metric < self.best_val_metric:
                        self.best_val_metric = current_val_metric
                        self.save(
                            metrics=val_metrics,
                            checkpoint_file="best_checkpoint.pt",
                            training_state=False,
                        )
                    self.model.train()

                self.scheduler_step(eval_every, current_val_metric)

                if is_final_batch:
                    break

                # End of batch.

            # End of epoch.
            epoch_times.append(time.time() - start_time)
            self.metrics["epoch_time"] = {"metric": epoch_times[-1]}
            self.log_train_metrics(end_of_epoch=True)
            torch.cuda.empty_cache()

        # End of training.

        if is_test_env:
            return

        eas = self.eval_all_splits(True, epoch=epoch_int, debug_batches=debug_batches)
        if eas == "SIGTERM":
            return "SIGTERM"

        if "test" in self.loaders:
            # TODO: update predict function
            # self.predict(self.loaders["test"], results_file="predictions")
            pass

        # Time model
        if self.logger is not None:
            log_epoch_times = False
            start_time = time.time()
            if self.config["optim"]["max_epochs"] == 0:
                batch = next(iter(self.loaders["train"]))
            else:
                log_epoch_times = True
            self.model_forward(batch)
            self.logger.log({"Batch time": time.time() - start_time})
            if log_epoch_times:
                self.logger.log({"Epoch time": sum(epoch_times) / len(epoch_times)})

        # Check respect of symmetries
        if self.test_ri and not is_test_env:
            symmetry = self.test_model_symmetries(debug_batches=debug_batches)
            if symmetry == "SIGTERM":
                return "SIGTERM"
            if self.logger:
                self.logger.log(symmetry)

        # TODO: Test equivariance

        # Close datasets
        if debug_batches < 0:
            for ds in self.datasets.values():
                ds.close_db()

    def model_forward(self, batch_list):
        # Distinguish frame averaging from base case.
        if self.config["frame_averaging"] and self.config["frame_averaging"] != "DA":
            original_pos = batch_list[0].pos
            if self.task_name in OCP_TASKS:
                original_cell = batch_list[0].cell
            e_all, p_all, f_all = [], [], []

            # Compute model prediction for each frame
            for i in range(len(batch_list[0].fa_pos)):
                batch_list[0].pos = batch_list[0].fa_pos[i]
                if self.task_name in OCP_TASKS:
                    batch_list[0].cell = batch_list[0].fa_cell[i]
                preds = self.model(deepcopy(batch_list))
                e_all.append(preds["energy"])
                if preds.get("pooling_loss") is not None:
                    p_all.append(preds["pooling_loss"])
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
            batch_list[0].pos = original_pos
            if self.task_name in OCP_TASKS:
                batch_list[0].cell = original_cell

            # Average predictions over frames
            preds = {"energy": sum(e_all) / len(e_all)}
            if len(p_all) > 0 and all(y is not None for y in p_all):
                preds["pooling_loss"] = sum(p_all) / len(p_all)
            if len(f_all) > 0 and all(y is not None for y in f_all):
                preds["forces"] = sum(f_all) / len(f_all)
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
                batch.y_relaxed.to(self.device)
                if self.task_name == "is2re"
                else batch.y.to(self.device)
                for batch in batch_list
            ],
            dim=0,
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
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
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    distutils.get_world_size() / train_loss_force_normalizer
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
                    energy_grad_mult = self.config["optim"].get(
                        "energy_grad_coefficient", 10
                    )
                    grad_target = preds["forces_grad_target"]
                    loss["energy_grad_loss"] = self.loss_fn["force"](
                        preds["forces"][mask], grad_target[mask]
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
                    batch.y_relaxed.to(self.device)
                    if self.task_name == "is2re"
                    else batch.y.to(self.device)
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
                and self.task_name in OCP_TASKS
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
                preds["forces"] = self.normalizers["grad_target"].denorm(
                    preds["forces"]
                )

        if self.normalizer.get("normalize_labels") and "target" in self.normalizers:
            preds["energy"] = self.normalizers["target"].denorm(preds["energy"])

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
            and distutils.is_master()
            and not self.is_hpo
        ) or (distutils.is_master() and end_of_epoch):
            if not self.silent:
                log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                print(
                    f"Train metrics at step {self.step}:\n  > " + "\n  > ".join(log_str)
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
        energy_diff_refl = torch.zeros(1, device=self.device)
        pos_diff_total = torch.zeros(1, device=self.device)
        forces_diff = torch.zeros(1, device=self.device)
        forces_diff_z = torch.zeros(1, device=self.device)
        forces_diff_refl = torch.zeros(1, device=self.device)

        for i, batch in enumerate(self.loaders[self.config["dataset"]["default_val"]]):
            if self.sigterm:
                return "SIGTERM"
            if debug_batches > 0 and i == debug_batches:
                break

            # Compute model prediction
            preds1 = self.model_forward(deepcopy(batch))

            # Compute prediction on rotated graph
            rotated = self.rotate_graph(batch, rotation="z")
            preds2 = self.model_forward(deepcopy(rotated["batch_list"]))

            # Difference in predictions, for energy and forces
            energy_diff_z += torch.abs(preds1["energy"] - preds2["energy"]).sum()
            if self.task_name == "s2ef":
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

            # Diff in positions
            pos_diff = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff = 0
                # Compute total difference across frames
                for pos1, pos2 in zip(batch[0].fa_pos, rotated["batch_list"][0].fa_pos):
                    pos_diff += pos1 - pos2
                # Manhanttan distance of pos matrix wrt 0 matrix.
                pos_diff_total += torch.abs(pos_diff).sum()

            # Reflect graph and compute diff in prediction
            reflected = self.reflect_graph(batch)
            preds3 = self.model_forward(reflected["batch_list"])
            energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()
            if self.task_name == "s2ef":
                forces_diff_refl += torch.abs(preds1["forces"] - preds3["forces"]).sum()

            # 3D Rotation and compute diff in prediction
            rotated = self.rotate_graph(batch)
            preds4 = self.model_forward(rotated["batch_list"])
            energy_diff += torch.abs(preds1["energy"] - preds4["energy"]).sum()
            if self.task_name == "s2ef":
                forces_diff += torch.abs(preds1["forces"] - preds4["forces"]).sum()

        # Aggregate the results
        batch_size = len(batch[0].natoms)
        energy_diff_z = energy_diff_z / (i * batch_size)
        energy_diff = energy_diff / (i * batch_size)
        energy_diff_refl = energy_diff_refl / (i * batch_size)
        pos_diff_total = pos_diff_total / (i * batch_size)

        symmetry = {
            "2D_E_ri": float(energy_diff_z),
            "3D_E_ri": float(energy_diff),
            "2D_pos_ri": float(pos_diff_total),
            "2D_E_refl_i": float(energy_diff_refl),
        }

        # Test equivariance of forces
        if self.task_name == "s2ef":
            forces_diff_z = forces_diff_z / (i * batch_size)
            forces_diff = forces_diff / (i * batch_size)
            forces_diff_refl = forces_diff_refl / (i * batch_size)
            symmetry.update(
                {
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
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
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
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
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

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()
