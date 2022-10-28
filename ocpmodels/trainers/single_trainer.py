"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import time
from copy import deepcopy

import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
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

        if self.normalizers is not None and "grad_target" in self.normalizers:
        if self.task_name == "s2ef":
            predictions["forces"] = []
            predictions["chunk_idx"] = []
            self.normalizers["grad_target"].to(self.device)
        for batch_list in tqdm(
        predictions = {"id": [], "energy": []}
        if self.task_name == "s2ef":
            predictions["forces"] = []
            predictions["chunk_idx"] = []

        for batch_list in tqdm(
            loader,
                out, _ = self._forward(batch_list)
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
            if self.normalizers is not None and "grad_target" in self.normalizers:
                self.normalizers["grad_target"].to(self.device)
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
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
            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
                if self.task_name == "s2ef":
                    predictions["forces"] = out["forces"].detach()
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
            "primary_metric", self.evaluator.task_primary_metric[self.name]
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
                        loss += pooling_loss
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
                            val_metrics[self.evaluator.task_primary_metric[self.name]][
                                "metric"
                            ]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
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
            y_all, p_all = [], []
            for i in range(len(batch_list[0].fa_pos)):
                batch_list[0].pos = batch_list[0].fa_pos[i]
                batch_list[0].cell = batch_list[0].fa_cell[i]
                y, p = self.model(deepcopy(batch_list))
                y_all.append(y)
                p_all.append(p)
            batch_list[0].pos = original_pos
            batch_list[0].cell = original_cell
            output = sum(y_all) / len(y_all)
            pooling_loss = sum(p_all) / len(p_all) if all(p_all) else None
        else:
            output, pooling_loss = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {
            "energy": output,
        }, pooling_loss

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.loss_fn["energy"](out["energy"], target_normed)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

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
            energies1, _ = self._forward(deepcopy(batch))

            # Rotate graph and compute prediction
            batch_rotated = self.rotate_graph(batch, rotation="z")
            energies2, _ = self._forward(deepcopy(batch_rotated))

            # Difference in predictions
            energy_diff_z += torch.abs(energies1["energy"] - energies2["energy"]).sum()

            # Diff in positions -- could remove model prediction
            pos_diff_z = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff_z = 0
                for pos1, pos2 in zip(batch[0].fa_pos, batch_rotated[0].fa_pos):
                    pos_diff_z += pos1 - pos2
                pos_diff_z = pos_diff_z.sum()

            # Reflect graph
            batch_reflected = self.reflect_graph(batch)
            energies3, _ = self._forward(batch_reflected)
            energy_diff_refl += torch.abs(
                energies1["energy"] - energies3["energy"]
            ).sum()

            # 3D Rotation
            batch_rotated = self.rotate_graph(batch)
            energies4, _ = self._forward(batch_rotated)
            energy_diff += torch.abs(energies1["energy"] - energies4["energy"]).sum()

        # Aggregate the results
        batch_size = len(batch[0].natoms)
        energy_diff_z = energy_diff_z / (i * batch_size)
        energy_diff = energy_diff / (i * batch_size)
        energy_diff_refl = energy_diff_refl / (i * batch_size)

        return energy_diff_z, energy_diff, pos_diff_z, energy_diff_refl
