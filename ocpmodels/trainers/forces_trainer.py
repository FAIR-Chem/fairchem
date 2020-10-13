import copy
import datetime
import json
import os

import torch
import torch_geometric
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.eval_relaxation import relax_eval
from ocpmodels.common.utils import plot_histogram, save_checkpoint
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("forces")
class ForcesTrainer(BaseTrainer):
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

        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if identifier:
            timestamp += "-{}".format(identifier)

        self.config = {
            "task": task,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp": timestamp,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", timestamp
                ),
                "results_dir": os.path.join(run_dir, "results", timestamp),
                "logs_dir": os.path.join(run_dir, "logs", logger, timestamp),
            },
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if isinstance(dataset, list):
            self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
        else:
            self.config["dataset"] = dataset

        if not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"])
            os.makedirs(self.config["cmd"]["results_dir"])
            os.makedirs(self.config["cmd"]["logs_dir"])

        self.is_debug = is_debug
        self.is_vis = is_vis
        if torch.cuda.is_available():
            self.device = local_rank
        else:
            self.device = "cpu"

        if distutils.is_master():
            print(yaml.dump(self.config, default_flow_style=False))
        self.load()

        self.evaluator = Evaluator(task="s2ef")

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        self.parallel_collater = ParallelCollater(1)
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

            if "relax_dataset" in self.config["task"]:
                self.relax_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["task"]["relax_dataset"])
                self.relax_loader = DataLoader(
                    self.relax_dataset,
                    batch_size=self.config["optim"].get("eval_batch_size", 64),
                    shuffle=False,
                    collate_fn=self.parallel_collater,
                    num_workers=self.config["optim"]["num_workers"],
                    pin_memory=True,
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
        if self.config["dataset"].get("normalize_labels", True):
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
            if self.config["dataset"].get("normalize_labels", True):
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

    def load_model(self):
        super(ForcesTrainer, self).load_model()

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1,
        )
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    # Takes in a new data source and generates predictions on it.
    def predict(self, dataset, batch_size=32, per_image=True):
        if isinstance(dataset, dict):
            if self.config["task"]["dataset"] == "trajectory_lmdb":
                print(
                    "### Generating predictions on {}.".format(dataset["src"])
                )
            else:
                print(
                    "### Generating predictions on {}.".format(
                        dataset["src"] + dataset["traj"]
                    )
                )

            dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(dataset)

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.parallel_collater,
            )
        elif isinstance(dataset, torch_geometric.data.Batch):
            data_loader = [[dataset]]
        else:
            raise NotImplementedError

        self.model.eval()
        predictions = {"energy": [], "forces": []}

        for i, batch_list in enumerate(data_loader):
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
                predictions["energy"].extend(out["energy"].tolist())
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                for natoms in batch_natoms:
                    predictions["forces"].append(
                        out["forces"][atoms_sum : natoms + atoms_sum]
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    atoms_sum += natoms
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                break

        return predictions

    def train(self):
        self.best_val_metric = -1.0
        eval_every = self.config["optim"].get("eval_every", -1)
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
                if i % self.config["cmd"]["print_every"] == 0:
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
                        val_metrics = self.validate(split="val", epoch=epoch)
                        if (
                            val_metrics[
                                self.config["task"].get(
                                    "primary_metric",
                                    self.evaluator.task_primary_metric["s2ef"],
                                )
                            ]["metric"]
                            > self.best_val_metric
                        ):
                            self.best_val_metric = val_metrics[
                                self.config["task"].get(
                                    "primary_metric",
                                    self.evaluator.task_primary_metric["s2ef"],
                                )
                            ]["metric"]
                            if not self.is_debug and distutils.is_master():
                                save_checkpoint(
                                    {
                                        "epoch": epoch
                                        + (i + 1) / len(self.train_loader),
                                        "state_dict": self.model.state_dict(),
                                        "optimizer": self.optimizer.state_dict(),
                                        "normalizers": {
                                            key: value.state_dict()
                                            for key, value in self.normalizers.items()
                                        },
                                        "config": self.config,
                                        "val_metrics": val_metrics,
                                    },
                                    self.config["cmd"]["checkpoint_dir"],
                                )

            self.scheduler.step()
            torch.cuda.empty_cache()

            if eval_every == -1:
                if self.val_loader is not None:
                    val_metrics = self.validate(split="val", epoch=epoch)
                    if (
                        val_metrics[
                            self.config["task"].get(
                                "primary_metric",
                                self.evaluator.task_primary_metric["s2ef"],
                            )
                        ]["metric"]
                        > self.best_val_metric
                    ):
                        self.best_val_metric = val_metrics[
                            self.config["task"].get(
                                "primary_metric",
                                self.evaluator.task_primary_metric["s2ef"],
                            )
                        ]["metric"]
                        if not self.is_debug and distutils.is_master():
                            save_checkpoint(
                                {
                                    "epoch": epoch + 1,
                                    "state_dict": self.model.state_dict(),
                                    "optimizer": self.optimizer.state_dict(),
                                    "normalizers": {
                                        key: value.state_dict()
                                        for key, value in self.normalizers.items()
                                    },
                                    "config": self.config,
                                    "val_metrics": val_metrics,
                                },
                                self.config["cmd"]["checkpoint_dir"],
                            )

            if self.test_loader is not None:
                self.validate(split="test", epoch=epoch)

        if "relax_dir" in self.config["task"]:
            self.validate_relaxation(
                split="val",
                epoch=epoch,
            )

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))

        self.model.eval()
        evaluator, metrics = Evaluator(task="s2ef"), {}

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

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
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": epoch + 1})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            print(", ".join(log_str))

        # Make plots.
        if self.logger is not None and epoch is not None:
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        return metrics

    def validate_relaxation(self, split="val", epoch=None):
        print("### Evaluating ML-relaxation")
        self.model.eval()

        evaluator, metrics = Evaluator(task="is2rs"), {}

        relaxed_positions = []
        for i, batch in enumerate(self.relax_loader):
            relaxed_batch = relax_eval(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                return_relaxed_pos=self.config["task"].get("write_pos", False),
                relax_opt=self.config["task"]["relax_opt"],
                device=self.device,
                transform=None,
            )

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
                ids = relaxed_batch.id
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]
                relaxed_positions += list(zip(ids.tolist(), batch_relaxed_positions))

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
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
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

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.json"
            )
            print("Writing relaxed pos to:", pos_filename)
            with open(pos_filename, "w") as f:
                json.dump(relaxed_positions, f)

        if distutils.is_master():
            print(metrics)

        return metrics

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
        if self.config["dataset"].get("normalize_labels", True):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(energy_mult * self.criterion(out["energy"], energy_target))

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.config["dataset"].get("normalize_labels", True):
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

        if self.config["dataset"].get("normalize_labels", True):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics
