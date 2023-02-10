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

from ocpmodels.common import dist_utils
from ocpmodels.common.registry import registry
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
        predictions = {"id": [], "energy": []}

        for batch in tqdm(
            loader,
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                preds = self.model_forward(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                preds["energy"] = self.normalizers["target"].denorm(preds["energy"])

            if per_image:
                predictions["id"].extend([str(i) for i in batch[0].sid.tolist()])
                predictions["energy"].extend(preds["energy"].tolist())
            else:
                predictions["energy"] = preds["energy"].detach()
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
                        loss += preds["pooling_loss"]

                loss = self.scaler.scale(loss) if self.scaler else loss
                if torch.isnan(loss):
                    print("\n\n >>> 🛑 Loss is NaN. Stopping training.\n\n")
                    self.logger.add_tags(["nan_loss"])
                    return True
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self.compute_metrics(
                    preds, batch, self.evaluator, metrics={}
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
                            self.eval_all_splits(is_final_epoch)

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

        # Check rotation invariance
        if self.test_ri and debug_batches < 0:
            (
                energy_diff_z,
                energy_diff,
                pos_diff_z,
                energy_diff_refl,
            ) = self.test_model_symmetries()
            if self.logger:
                self.logger.log({"2D_ri": energy_diff_z})
                self.logger.log({"3D_ri": energy_diff})
                self.logger.log({"2D_pos_ri": pos_diff_z})
                self.logger.log({"2D_pos_refl_i": energy_diff_refl})

        # TODO: Test equivariance

        # Evaluate current model on all 4 validation splits
        # self.eval_all_splits()

        # Close datasets
        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def model_forward(self, batch_list):

        if self.config["frame_averaging"] and self.config["frame_averaging"] != "DA":
            original_pos = batch_list[0].pos
            original_cell = batch_list[0].cell
            y_all, p_all = [], []
            for i in range(len(batch_list[0].fa_pos)):
                batch_list[0].pos = batch_list[0].fa_pos[i]
                batch_list[0].cell = batch_list[0].fa_cell[i]
                preds = self.model(deepcopy(batch_list))
                y_all.append(preds["energy"])
                if preds.get("pooling_loss") is not None:
                    p_all.append(preds["pooling_loss"])
            batch_list[0].pos = original_pos
            batch_list[0].cell = original_cell
            preds["energy"] = sum(y_all) / len(y_all)
            preds["pooling_loss"] = (
                sum(p_all) / len(p_all)
                if (p_all and all(y is not None for y in p_all))
                else None
            )
        else:
            preds = self.model(batch_list)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"].view(-1)

        return preds

    def compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.loss_fn["energy"](out["energy"], target_normed)
        return loss

    def compute_metrics(self, out, batch_list, evaluator, metrics={}):
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
            and dist_utils.is_master()
            and not self.is_hpo
        ) or (dist_utils.is_master() and end_of_epoch):
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
            preds1 = self.model_forward(deepcopy(batch))

            # Rotate graph and compute prediction
            rotated = self.rotate_graph(batch, rotation="z")
            preds2 = self.model_forward(deepcopy(rotated["batch_list"]))

            # Difference in predictions
            energy_diff_z += torch.abs(preds1["energy"] - preds2["energy"]).sum()

            # Diff in positions -- could remove model prediction
            pos_diff_z = -1
            if hasattr(batch[0], "fa_pos"):
                pos_diff_z = 0
                for pos1, pos2 in zip(batch[0].fa_pos, rotated["batch_list"][0].fa_pos):
                    pos_diff_z += pos1 - pos2
                pos_diff_z = pos_diff_z.sum()

            # Reflect graph
            batch_reflected = self.reflect_graph(batch)
            preds3 = self.model_forward(batch_reflected)
            energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()

            # 3D Rotation
            rotated = self.rotate_graph(batch)
            preds4 = self.model_forward(rotated["batch_list"])
            energy_diff += torch.abs(preds1["energy"] - preds4["energy"]).sum()

        # Aggregate the results
        batch_size = len(batch[0].natoms)
        energy_diff_z = energy_diff_z / (i * batch_size)
        energy_diff = energy_diff / (i * batch_size)
        energy_diff_refl = energy_diff_refl / (i * batch_size)

        return energy_diff_z, energy_diff, pos_diff_z, energy_diff_refl
