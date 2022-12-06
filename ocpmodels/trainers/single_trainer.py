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

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import OCP_TASKS, check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


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
                        tensor=self.train_loader.dataset.data.y[
                            self.train_loader.dataset.__indices__
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

    def train(self, disable_eval_tqdm=False, debug_batches=-1):
        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        self.config["print_every"] = eval_every  # Can comment out for better debug
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.task_name]
        )
        self.best_val_mae = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        epoch_time = []

        if not self.silent:
            print("---Beginning of Training---")

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):

            start_time = time.time()
            if not self.silent:
                print("Epoch: ", epoch_int)

            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                if debug_batches > 0 and i == debug_batches:
                    break

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    preds = self.model_forward(batch)
                    loss = self.compute_loss(preds, batch)
                    if preds.get("pooling_loss") is not None:
                        coeff = self.config["optim"].get("pooling_coefficient", 1)
                        loss += preds["pooling_loss"] * coeff
                loss = self.scaler.scale(loss) if self.scaler else loss
                if torch.isnan(loss):
                    print("\n\n >>> 🛑 Loss is NaN. Stopping training.\n\n")
                    self.logger.add_tags(["nan_loss"])
                    return True
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self.compute_metrics(
                    preds,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                self._log_metrics()

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file=f"checkpoint-{str(self.step).zfill(6)}.pt",
                        training_state=True,
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.task_name]
                            ]["metric"]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.task_name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

                        # Evaluate current model on all 4 validation splits
                        is_final_epoch = debug_batches < 0 and (
                            epoch_int == self.config["optim"]["max_epochs"] - 1
                        )
                        if (epoch_int % 100 == 0 and epoch_int != 0) or is_final_epoch:
                            self.eval_all_val_splits(is_final_epoch, epoch=epoch_int)

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

                # End of batch.

            # End of epoch.
            self._log_metrics(end_of_epoch=True)
            torch.cuda.empty_cache()
            epoch_time.append(time.time() - start_time)

        # End of training.

        # Time model
        if self.logger is not None:
            start_time = time.time()
            # batch = next(iter(self.train_loader))
            self.model_forward(batch)
            self.logger.log({"Batch time": time.time() - start_time})
            self.logger.log({"Epoch time": sum(epoch_time) / len(epoch_time)})

        # Check respect of symmetries
        if self.test_ri and debug_batches < 0:
            symmetry = self.test_model_symmetries()
            if self.logger:
                self.logger.log(symmetry)

        # TODO: Test equivariance

        # Evaluate current model on all 4 validation splits
        # self.eval_all_val_splits()

        # Close datasets
        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

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
        loss = []

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
        energy_mult = (
            self.config["optim"].get("energy_coefficient", 1)
            if self.task_name in {"is2rs", "s2ef"}
            else 1
        )
        loss.append(
            energy_mult * self.loss_fn["energy"](preds["energy"], target_normed)
        )

        # Force loss.
        if self.task_name in {"is2rs", "s2ef"}:
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

                loss.append(
                    force_mult
                    * self.loss_fn["force"](preds["forces"][mask], force_target[mask])
                )
                if "forces_grad_target" in preds:
                    energy_grad_mult = self.config["optim"].get(
                        "energy_grad_coefficient", 10
                    )
                    grad_target = preds["forces_grad_target"]
                    loss.append(
                        energy_grad_mult
                        * self.loss_fn["force"](
                            preds["forces"][mask], grad_target[mask]
                        )
                    )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def compute_metrics(self, preds, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [
                    (
                        batch.y.to(self.device)
                        if self.task_name in {"s2ef", "qm9"}
                        else batch.y_relaxed.to(self.device)
                    )
                    for batch in batch_list
                ],
                dim=0,
            ),
            "natoms": natoms,
        }

        if self.task_name == "s2ef":
            target["forces"] = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            preds["natoms"] = natoms

            if "forces_grad_target" in preds:
                target["forces_grad_target"] = preds["forces_grad_target"]

            if self.config["task"].get("eval_on_free_atoms", True):
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

    def _log_metrics(self, end_of_epoch=False):
        log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
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
            log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
            if not self.silent:
                print(", ".join(log_str))
            self.metrics = {}

        if self.logger is not None and not end_of_epoch:
            self.logger.log(
                log_dict,
                step=self.step,
                split="train",
            )

    @torch.no_grad()
    def test_model_symmetries(self, debug_batches=100):
        """Test rotation and reflection invariance & equivariance
        of GNNs

        Returns:
            (tensors): metrics to measure RI difference in
            energy/force pred. or pos. between G and rotated G
        """

        self.model.eval()

        energy_diff = torch.zeros(1, device=self.device)
        energy_diff_z = torch.zeros(1, device=self.device)
        energy_diff_refl = torch.zeros(1, device=self.device)
        forces_diff = torch.zeros(1, device=self.device)
        forces_diff_z = torch.zeros(1, device=self.device)
        forces_diff_refl = torch.zeros(1, device=self.device)

        for i, batch in enumerate(self.val_loader):
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
            pos_diff_z = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff_z = 0
                for pos1, pos2 in zip(batch[0].fa_pos, rotated["batch_list"][0].fa_pos):
                    pos_diff_z += pos1 - pos2
                pos_diff_z = pos_diff_z.sum()

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

        symmetry = {
            "2D_E_ri": energy_diff_z,
            "3D_E_ri": energy_diff,
            "2D_pos_ri": pos_diff_z,
            "2D_E_refl_i": energy_diff_refl,
        }

        # Test equivariance of forces
        if self.task_name == "s2ef":
            forces_diff_z = forces_diff_z / (i * batch_size)
            forces_diff = forces_diff / (i * batch_size)
            forces_diff_refl = forces_diff_refl / (i * batch_size)
            symmetry.update(
                {
                    "2D_F_ri": forces_diff_z,
                    "3D_F_ri": forces_diff,
                    "2D_F_refl_i": forces_diff_refl,
                }
            )

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
