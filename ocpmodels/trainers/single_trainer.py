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
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("energy")
class EnergyTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_. # noqa: E501
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, name="is2re")

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
        if self.config["model"].get("regress_forces", True):
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
                out, _ = self._forward(batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
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
                predictions["energy"].extend(out["energy"].to(torch.float16).tolist())

                if self.task_name == "s2ef":
                    batch_natoms = torch.cat([batch.natoms for batch in batch_list])
                    batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                    forces = out["forces"].cpu().detach().to(torch.float16)
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
                predictions["energy"] = out["energy"].detach()
                if self.task_name == "s2ef":
                    predictions["forces"] = out["forces"].detach()
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
                    out, pooling_loss = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                    if pooling_loss is not None:
                        loss += pooling_loss * self.config["optim"].get(
                            "pooling_coefficient", 1
                        )
                loss = self.scaler.scale(loss) if self.scaler else loss
                if torch.isnan(loss):
                    print("\n\n >>> 🛑 Loss is NaN. Stopping training.\n\n")
                    self.logger.add_tags(["nan_loss"])
                    return True
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
                            self.eval_all_val_splits(is_final_epoch)

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
            self._forward(batch)
            self.logger.log({"Batch time": time.time() - start_time})
            self.logger.log({"Epoch time": sum(epoch_time) / len(epoch_time)})

        # Check rotation invariance
        if self.test_ri and debug_batches < 0:
            (
                energy_diff_z,
                energy_diff,
                pos_diff_z,
                energy_diff_refl,
            ) = self.test_model_invariance()
            if self.logger:
                self.logger.log({"2D_ri": energy_diff_z})
                self.logger.log({"3D_ri": energy_diff})
                self.logger.log({"2D_pos_ri": pos_diff_z})
                self.logger.log({"2D_pos_refl_i": energy_diff_refl})

        # TODO: Test equivariance

        # Evaluate current model on all 4 validation splits
        # self.eval_all_val_splits()

        # Close datasets
        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def _forward(self, batch_list):

        if self.config["frame_averaging"] and self.config["frame_averaging"] != "DA":
            original_pos = batch_list[0].pos
            original_cell = batch_list[0].cell
            e_all, p_all, f_all = [], [], []

            for i in range(len(batch_list[0].fa_pos)):
                batch_list[0].pos = batch_list[0].fa_pos[i]
                batch_list[0].cell = batch_list[0].fa_cell[i]
                e, p, *f = self.model(deepcopy(batch_list))
                e_all.append(e)
                p_all.append(p)
                if f:
                    f_all.append(f[0])

            batch_list[0].pos = original_pos
            batch_list[0].cell = original_cell
            energy = sum(e_all) / len(e_all)
            forces = [sum(f_all) / len(f_all)] if f_all else []
            pooling_loss = sum(p_all) / len(p_all) if all(p_all) else None
        else:
            energy, pooling_loss, *forces = self.model(batch_list)

        if energy.shape[-1] == 1:
            energy = energy.view(-1)

        out = {"energy": energy}

        if forces:
            out["forces"] = forces[0]

        return out, pooling_loss

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(energy_mult * self.loss_fn["energy"](out["energy"], target_normed))

        # Force loss.
        if self.config["model"].get("regress_forces", True):
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

                loss_force_list = torch.abs(out["forces"] - force_target)
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
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"][mask], force_target[mask])
                    )
                else:
                    loss.append(
                        force_mult * self.loss_fn["force"](out["forces"], force_target)
                    )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        if self.task_name == "s2ef":
            target["forces"] = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            out["natoms"] = natoms

            if self.config["task"].get("eval_on_free_atoms", True):
                fixed = torch.cat([batch.fixed.to(self.device) for batch in batch_list])
                mask = fixed == 0
                out["forces"] = out["forces"][mask]
                target["forces"] = target["forces"][mask]

                s_idx = 0
                natoms_free = []
                for natoms in target["natoms"]:
                    natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                    s_idx += natoms
                target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
                out["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            if (
                self.normalizer.get("normalize_labels")
                and "grad_target" in self.normalizers
            ):
                out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])

        if self.normalizer.get("normalize_labels") and "target" in self.normalizers:
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(out, target, prev_metrics=metrics)

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
    def test_model_invariance(self, debug_batches=100):
        """Test rotation and reflection invariance properties of GNNs

        Returns:
            (tensor, tensor, tensor): metrics to measure RI
            difference in energy pred. / pos. between G and rotated G
        """

        self.model.eval()
        energy_diff = torch.zeros(1, device=self.device)
        energy_diff_z = torch.zeros(1, device=self.device)
        energy_diff_refl = torch.zeros(1, device=self.device)

        for i, batch in enumerate(self.val_loader):
            if debug_batches > 0 and i == debug_batches:
                break
            # Pass it through the model.
            out1, _ = self._forward(deepcopy(batch))

            # Rotate graph and compute prediction
            batch_rotated = self.rotate_graph(batch, rotation="z")
            out2, _ = self._forward(deepcopy(batch_rotated))

            # Difference in predictions
            energy_diff_z += torch.abs(out1["energy"] - out2["energy"]).sum()

            # Diff in positions -- could remove model prediction
            pos_diff_z = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff_z = 0
                for pos1, pos2 in zip(batch[0].fa_pos, batch_rotated[0].fa_pos):
                    pos_diff_z += pos1 - pos2
                pos_diff_z = pos_diff_z.sum()

            # Reflect graph
            batch_reflected = self.reflect_graph(batch)
            out3, _ = self._forward(batch_reflected)
            energy_diff_refl += torch.abs(out1["energy"] - out3["energy"]).sum()

            # 3D Rotation
            batch_rotated = self.rotate_graph(batch)
            out4, _ = self._forward(batch_rotated)
            energy_diff += torch.abs(out1["energy"] - out4["energy"]).sum()

        # Aggregate the results
        batch_size = len(batch[0].natoms)
        energy_diff_z = energy_diff_z / (i * batch_size)
        energy_diff = energy_diff / (i * batch_size)
        energy_diff_refl = energy_diff_refl / (i * batch_size)

        return energy_diff_z, energy_diff, pos_diff_z, energy_diff_refl

    def run_relaxations(self, split="val"):
        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

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
