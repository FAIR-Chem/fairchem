"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch_geometric
from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import plot_histogram
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, DistributedSampler


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
        is_vis (bool, optional): Run in debug mode.
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
        local_rank=0,
        amp=False,
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
            local_rank=local_rank,
            amp=amp,
            name="s2ef",
        )

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        self.parallel_collater = ParallelCollater(
            1, self.config["model_attributes"].get("otf_graph", False)
        )
        if self.config["task"]["dataset"] == "trajectory_lmdb":
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config["optim"]["batch_size"],
                shuffle=True,
                collate_fn=self.parallel_collater,
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=True,
            )

            self.val_loader = self.test_loader = None

            if "val_dataset" in self.config:
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["val_dataset"])
                self.val_loader = DataLoader(
                    self.val_dataset,
                    self.config["optim"].get("eval_batch_size", 64),
                    shuffle=False,
                    collate_fn=self.parallel_collater,
                    num_workers=self.config["optim"]["num_workers"],
                    pin_memory=True,
                )
            if "test_dataset" in self.config:
                self.test_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["test_dataset"])
                self.test_loader = DataLoader(
                    self.test_dataset,
                    self.config["optim"].get("eval_batch_size", 64),
                    shuffle=False,
                    collate_fn=self.parallel_collater,
                    num_workers=self.config["optim"]["num_workers"],
                    pin_memory=True,
                )

            if "relax_dataset" in self.config["task"]:
                assert os.path.isfile(
                    self.config["task"]["relax_dataset"]["src"]
                )

                self.relax_dataset = registry.get_dataset_class(
                    "single_point_lmdb"
                )(self.config["task"]["relax_dataset"])

                self.relax_sampler = DistributedSampler(
                    self.relax_dataset,
                    num_replicas=distutils.get_world_size(),
                    rank=distutils.get_rank(),
                    shuffle=False,
                )
                self.relax_loader = DataLoader(
                    self.relax_dataset,
                    batch_size=self.config["optim"].get("eval_batch_size", 64),
                    collate_fn=self.parallel_collater,
                    num_workers=self.config["optim"]["num_workers"],
                    pin_memory=True,
                    sampler=self.relax_sampler,
                )

        else:
            self.dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
            ) = self.dataset.get_dataloaders(
                batch_size=self.config["optim"]["batch_size"],
                collate_fn=self.parallel_collater,
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
                self.normalizers["target"] = Normalizer(
                    tensor=self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    device=self.device,
                )

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config["model_attributes"].get("regress_forces", True):
            if self.config["dataset"].get("normalize_labels", False):
                if "grad_target_mean" in self.config["dataset"]:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.config["dataset"]["grad_target_mean"],
                        std=self.config["dataset"]["grad_target_std"],
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

        if (
            self.is_vis
            and self.config["task"]["dataset"] != "qm9"
            and distutils.is_master()
        ):
            # Plot label distribution.
            plots = [
                plot_histogram(
                    self.train_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: train",
                ),
                plot_histogram(
                    self.val_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: val",
                ),
                plot_histogram(
                    self.test_loader.dataset.data.y.tolist(),
                    xlabel="{}/raw".format(self.config["task"]["labels"][0]),
                    ylabel="# Examples",
                    title="Split: test",
                ),
            ]
            self.logger.log_plots(plots)

    # Takes in a new data source and generates predictions on it.
    def predict(
        self, data_loader, per_image=True, results_file=None, disable_tqdm=True
    ):
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
        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": []}

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
                atoms_sum = 0
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                predictions["energy"].extend(out["energy"].tolist())
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                for natoms in batch_natoms:
                    forces = (
                        out["forces"][atoms_sum : natoms + atoms_sum]
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    # evalAI only requires forces on free atoms
                    if results_file is not None:
                        _free_atoms = (
                            batch_fixed[atoms_sum : natoms + atoms_sum] == 0
                        ).tolist()
                        forces = forces[_free_atoms]
                    atoms_sum += natoms
                    predictions["forces"].append(forces)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                break

        self.save_results(predictions, results_file, keys=["energy", "forces"])
        return predictions

    def train(self):
        self.best_val_metric = -1.0
        eval_every = self.config["optim"].get("eval_every", -1)
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        iters = 0
        self.metrics = {}
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
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
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                iters += 1

                # Evaluate on val set every `eval_every` iterations.
                if eval_every != -1 and iters % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            epoch=epoch - 1 + (i + 1) / len(self.train_loader),
                        )
                        if (
                            val_metrics[primary_metric]["metric"]
                            > self.best_val_metric
                        ):
                            self.best_val_metric = val_metrics[primary_metric][
                                "metric"
                            ]
                            current_epoch = epoch + (i + 1) / len(
                                self.train_loader
                            )
                            self.save(current_epoch, val_metrics)

            self.scheduler.step()
            torch.cuda.empty_cache()

            if eval_every == -1:
                if self.val_loader is not None:
                    val_metrics = self.validate(split="val", epoch=epoch)
                    if (
                        val_metrics[primary_metric]["metric"]
                        > self.best_val_metric
                    ):
                        self.best_val_metric = val_metrics[primary_metric][
                            "metric"
                        ]
                        self.save(epoch + 1, val_metrics)
                else:
                    self.save(epoch + 1, self.metrics)

            if self.test_loader is not None:
                self.validate(split="test", epoch=epoch)

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

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.config["dataset"].get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(energy_mult * self.criterion(out["energy"], energy_target))

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.config["dataset"].get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            # Force coefficient = 30 has been working well for us.
            force_mult = self.config["optim"].get("force_coefficient", 30)
            if self.config["task"].get("train_on_free_atoms", False):
                fixed = torch.cat(
                    [batch.fixed.to(self.device) for batch in batch_list]
                )
                mask = fixed == 0
                loss.append(
                    force_mult
                    * self.criterion(out["forces"][mask], force_target[mask])
                )
            else:
                loss.append(
                    force_mult * self.criterion(out["forces"], force_target)
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

        if self.config["dataset"].get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def run_relaxations(self, split="val", epoch=None):
        print("### Running ML-relaxations")
        self.model.eval()

        evaluator, metrics = Evaluator(task="is2rs"), {}

        if hasattr(self.relax_dataset[0], "pos_relaxed") and hasattr(
            self.relax_dataset[0], "y_relaxed"
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
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

                metrics = evaluator.eval(prediction, target, metrics)

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez(pos_filename, ids=ids, pos=relaxed_positions)

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
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = gather_results["ids"][idx]
                gather_results["pos"] = gather_results["pos"][idx]

                print(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
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
                    aggregated_metrics[k]["total"]
                    / aggregated_metrics[k]["numel"]
                )
            metrics = aggregated_metrics

            # Make plots.
            log_dict = {k: metrics[k]["metric"] for k in metrics}
            if self.logger is not None and epoch is not None:
                self.logger.log(
                    log_dict,
                    step=(epoch + 1) * len(self.train_loader),
                    split=split,
                )

            if distutils.is_master():
                print(metrics)
