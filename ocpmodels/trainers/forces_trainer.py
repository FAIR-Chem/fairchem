"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("forces")
class ForcesTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

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
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed: Optional[int] = None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        slurm={},
        noddp: bool = False,
    ) -> None:
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="s2ef",
            slurm=slurm,
            noddp=noddp,
        )

    def load_task(self) -> None:
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")

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

        self.num_targets = 1

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config["model_attributes"].get("regress_forces", True):
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

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file=None,
        disable_tqdm: bool = False,
    ) -> Dict[str, npt.NDArray[np.float_]]:
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
                out["forces"] = self.normalizers["grad_target"].denorm(
                    out["forces"]
                )
            if per_image:
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                # total energy target requires predictions to be saved in float32
                # default is float16
                if (
                    self.config["task"].get("prediction_dtype", "float16")
                    == "float32"
                    or self.config["task"]["dataset"] == "oc22_lmdb"
                ):
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float32).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float32)
                else:
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float16).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_forces, _per_image_fixed
                        )
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
                predictions["forces"] = out["forces"].detach()
                if self.ema:
                    self.ema.restore()
                return predictions

        predictions["forces"] = np.array(predictions["forces"], dtype=object)
        predictions["chunk_idx"] = np.array(
            predictions["chunk_idx"],
        )
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(
            predictions["id"],
        )
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )

        if self.ema:
            self.ema.restore()

        return predictions

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

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
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
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )
                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces = self.model(batch_list)
        else:
            out_energy = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        return out

    def _compute_loss(self, out, batch_list) -> int:
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(
            energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                if self.config["optim"].get("loss_force", "l2mae") == "l2mae":
                    # zero out nans, if any
                    found_nans_or_infs = not torch.all(
                        out["forces"].isfinite()
                    )
                    if found_nans_or_infs is True:
                        logging.warning("Found nans while computing loss")
                        out["forces"] = torch.nan_to_num(
                            out["forces"], nan=0.0
                        )

                    dists = torch.norm(
                        out["forces"] - force_target, p=2, dim=-1
                    )
                    weighted_dists_sum = (dists * weight).sum()

                    num_samples = out["forces"].shape[0]
                    num_samples = distutils.all_reduce(
                        num_samples, device=self.device
                    )
                    weighted_dists_sum = (
                        weighted_dists_sum
                        * distutils.get_world_size()
                        / num_samples
                    )

                    force_mult = self.config["optim"].get(
                        "force_coefficient", 30
                    )
                    loss.append(force_mult * weighted_dists_sum)
                else:
                    raise NotImplementedError
            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    if (
                        self.config["optim"]
                        .get("loss_force", "mae")
                        .startswith("atomwise")
                    ):
                        force_mult = self.config["optim"].get(
                            "force_coefficient", 1
                        )
                        natoms = torch.cat(
                            [
                                batch.natoms.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            out["forces"][mask],
                            force_target[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                out["forces"][mask], force_target[mask]
                            )
                        )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
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
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(
                    torch.sum(mask[s_idx : s_idx + natoms]).item()
                )
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def run_relaxations(self, split: str = "val") -> None:
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
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
                save_full_traj=self.config["task"].get("save_full_traj", True),
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
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
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
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
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
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
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
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
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

        registry.unregister("set_deterministic_scatter")
