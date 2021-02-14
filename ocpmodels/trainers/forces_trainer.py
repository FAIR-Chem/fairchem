"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import plot_histogram
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
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
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="s2ef",
        )

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        self.parallel_collater = ParallelCollater(
            1 if not self.cpu else 0,
            self.config["model_attributes"].get("otf_graph", False),
        )
        if self.config["task"]["dataset"] == "trajectory_lmdb":
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            print(len(self.train_dataset))

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
        if self.config["model_attributes"].get("regress_relaxed_energy", True):
            self.num_targets += 1
        if self.config["model_attributes"].get(
            "regress_relaxed_position", True
        ):
            self.num_targets += 3

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

        if self.config["model_attributes"].get("regress_relaxed_energy", True):
            if self.config["dataset"].get("normalize_labels", False):
                if "target_relaxed_energy_mean" in self.config["dataset"]:
                    self.normalizers["target_relaxed_energy"] = Normalizer(
                        mean=self.config["dataset"][
                            "target_relaxed_energy_mean"
                        ],
                        std=self.config["dataset"][
                            "target_relaxed_energy_std"
                        ],
                        device=self.device,
                    )
                else:
                    self.normalizers["target_relaxed_energy"] = Normalizer(
                        tensor=self.train_loader.dataset.data.relaxed_y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )
        if self.config["model_attributes"].get(
            "regress_relaxed_position", True
        ):
            if self.config["dataset"].get("normalize_labels", False):
                if "target_relaxed_pos_mean" in self.config["dataset"]:
                    self.normalizers["target_relaxed_position"] = Normalizer(
                        mean=self.config["dataset"]["target_relaxed_pos_mean"],
                        std=self.config["dataset"]["target_relaxed_pos_std"],
                        device=self.device,
                    )
                else:
                    self.normalizers["target_relaxed_position"] = Normalizer(
                        tensor=self.train_loader.dataset.data.relaxed_y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )

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
        if distutils.is_master() and not disable_tqdm:
            print("### Predicting on test.")
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

        if "target_relaxed_energy" in self.normalizers:
            self.normalizers["target_relaxed_energy"].to(self.device)
            predictions["relaxed_energy"] = []

        if "target_relaxed_position" in self.normalizers:
            self.normalizers["target_relaxed_position"].to(self.device)
            predictions["positions"] = []

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
            if "target_relaxed_energy" in self.normalizers:
                out["target_relaxed_energy"] = self.normalizers[
                    "target_relaxed_energy"
                ].denorm(out["target_relaxed_energy"])
            if "target_relaxed_position" in self.normalizers:
                out["target_relaxed_position"] = self.normalizers[
                    "target_relaxed_position"
                ].denorm(out["target_relaxed_position"])
            if per_image:
                atoms_sum = 0
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                predictions["energy"].extend(
                    out["energy"].to(torch.float16).tolist()
                )
                if "target_relaxed_energy" in self.normalizers:
                    predictions["relaxed_energy"].extend(
                        out["relaxed_energy"].to(torch.float16).tolist()
                    )
                if "target_relaxed_pos" in self.normalizers:
                    predictions["positions"].extend(
                        out["positions"].to(torch.float16).tolist()
                    )
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                for natoms in batch_natoms:
                    forces = (
                        out["forces"][atoms_sum : natoms + atoms_sum]
                        .cpu()
                        .detach()
                        .to(torch.float16)
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
                if "target_relaxed_energy" in self.normalizers:
                    predictions["relaxed_energy"] = out[
                        "relaxed_energy"
                    ].detach()
                if "target_relaxed_position" in self.normalizers:
                    predictions["positions"] = out["positions"].detach()
                return predictions

        predictions["id"] = np.array(predictions["id"])
        predictions["forces"] = np.array(predictions["forces"], dtype=object)
        predictions["energy"] = np.array(predictions["energy"])
        keys = ["energy", "forces"]
        if "target_relaxed_energy" in self.normalizers:
            predictions["relaxed_energy"] = np.array(
                predictions["relaxed_energy"]
            )
            keys.append("target_relaxed_energy")
        if "target_relaxed_position" in self.normalizers:
            predictions["positions"] = np.array(predictions["positions"])
            keys.append("target_relaxed_position")
        self.save_results(predictions, results_file, keys=keys)
        return predictions

    def get_mean_stddev_relaxed_pos(self):

        all_relaxed_pos = []
        for data in self.train_dataset:
            relaxed_force = np.array(data.relaxed_force)
            if np.amax(np.abs(relaxed_force)) <= 0.05:
                all_relaxed_pos.extend(data.relaxed_pos.tolist())
        all_relaxed_pos = np.array(all_relaxed_pos)
        print(np.mean(all_relaxed_pos), np.std(all_relaxed_pos))

    def train(self):
        eval_every = self.config["optim"].get("eval_every", -1)
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
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
                            "mae" in primary_metric
                            and val_metrics[primary_metric]["metric"]
                            < self.best_val_metric
                        ) or (
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
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

            self.scheduler.step()
            torch.cuda.empty_cache()

            if eval_every == -1:
                if self.val_loader is not None:
                    val_metrics = self.validate(split="val", epoch=epoch)

                    if (
                        "mae" in primary_metric
                        and val_metrics[primary_metric]["metric"]
                        < self.best_val_metric
                    ) or (
                        val_metrics[primary_metric]["metric"]
                        > self.best_val_metric
                    ):
                        self.best_val_metric = val_metrics[primary_metric][
                            "metric"
                        ]
                        self.save(epoch + 1, val_metrics)
                        if self.test_loader is not None:
                            self.predict(
                                self.test_loader,
                                results_file="predictions",
                                disable_tqdm=False,
                            )
                else:
                    self.save(epoch + 1, self.metrics)

        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        # forward pass.
        outputs = self.model(batch_list)
        regress_forces = self.config["model_attributes"].get(
            "regress_forces", True
        )
        regress_relaxed_energy = self.config["model_attributes"].get(
            "regress_relaxed_energy", True
        )
        regress_relaxed_pos = self.config["model_attributes"].get(
            "regress_relaxed_position", True
        )

        if regress_forces:
            if regress_relaxed_pos:
                out_energy, out_forces, out_relaxed_pos = outputs
            else:
                out_energy, out_forces = outputs
        else:
            if regress_relaxed_pos:
                out_energy, out_relaxed_pos = outputs
            else:
                out_energy = outputs

        out_energy_ = out_energy[:, 0]
        if regress_relaxed_energy:
            out_relaxed_energy = out_energy[:, 1:2]

        if out_energy_.shape[-1] == 1:
            out_energy_ = out_energy_.view(-1)

        out = {
            "energy": out_energy_,
        }

        if regress_forces:
            out["forces"] = out_forces

        if regress_relaxed_energy:
            if out_relaxed_energy.shape[-1] == 1:
                out_relaxed_energy = out_relaxed_energy.view(-1)
            out["relaxed_energy"] = out_relaxed_energy

        if regress_relaxed_pos:
            out["positions"] = out_relaxed_pos

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
                        * self.criterion(
                            out["forces"][mask], force_target[mask]
                        )
                    )
                else:
                    loss.append(
                        force_mult
                        * self.criterion(out["forces"], force_target)
                    )

        if self.config["model_attributes"].get("regress_relaxed_energy", True):
            relaxed_energy_target = torch.cat(
                [batch.relaxed_y.to(self.device) for batch in batch_list],
                dim=0,
            )
            if self.config["dataset"].get("normalize_labels", False):
                relaxed_energy_target = self.normalizers[
                    "target_relaxed_energy"
                ].norm(relaxed_energy_target)
            relaxed_energy_mult = self.config["optim"].get(
                "relaxed_energy_coefficient", 1
            )
            # converged = torch.cat(
            #         [batch.converged for batch in batch_list]
            #     )
            # loss.append(
            #     relaxed_energy_mult
            #     * self.criterion(out["relaxed_energy"][converged], relaxed_energy_target[converged])
            # )
            loss.append(
                relaxed_energy_mult
                * self.criterion(out["relaxed_energy"], relaxed_energy_target)
            )

        if self.config["model_attributes"].get(
            "regress_relaxed_position", True
        ):
            relaxed_pos_target = torch.cat(
                [batch.relaxed_pos.to(self.device) for batch in batch_list],
                dim=0,
            )
            if self.config["dataset"].get("normalize_labels", False):
                relaxed_pos_target = self.normalizers[
                    "target_relaxed_position"
                ].norm(relaxed_pos_target)
            relaxed_pos_mult = self.config["optim"].get(
                "relaxed_pos_coefficient", 1
            )
            if self.config["task"].get("train_on_free_atoms", False):
                fixed = torch.cat(
                    [batch.fixed.to(self.device) for batch in batch_list]
                )
                mask = fixed == 0
                loss.append(
                    relaxed_pos_mult
                    * self.criterion(
                        out["positions"][mask], relaxed_pos_target[mask]
                    )
                )
            else:
                loss.append(
                    relaxed_pos_mult
                    * self.criterion(out["positions"], relaxed_pos_target)
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

        if self.config["model_attributes"].get("regress_relaxed_energy", True):
            target["relaxed_energy"] = torch.cat(
                [batch.relaxed_y.to(self.device) for batch in batch_list],
                dim=0,
            )

        if self.config["model_attributes"].get(
            "regress_relaxed_position", True
        ):
            target["positions"] = torch.cat(
                [batch.relaxed_pos.to(self.device) for batch in batch_list],
                dim=0,
            )

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            if self.config["model_attributes"].get(
                "regress_relaxed_position", True
            ):
                out["positions"] = out["positions"][mask]
                target["positions"] = target["positions"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(
                    torch.sum(mask[s_idx : s_idx + natoms]).item()
                )
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.config["model_attributes"].get(
            "regress_relaxed_position", True
        ):
            cell = torch.cat(
                [batch.cell.to(self.device) for batch in batch_list]
            )
            target["cell"] = cell
            out["cell"] = cell

            pbc = torch.tensor([True, True, True])
            target["pbc"] = pbc
            out["pbc"] = pbc

        if self.config["dataset"].get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

            if self.config["model_attributes"].get(
                "regress_relaxed_energy", True
            ):
                out["relaxed_energy"] = self.normalizers[
                    "target_relaxed_energy"
                ].denorm(out["relaxed_energy"])

            if self.config["model_attributes"].get(
                "regress_relaxed_position", True
            ):
                out["positions"] = self.normalizers[
                    "target_relaxed_position"
                ].denorm(out["positions"])

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
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
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
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.array(
                    gather_results["pos"], dtype=object
                )[idx]

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
