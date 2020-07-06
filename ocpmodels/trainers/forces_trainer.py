import datetime
import os

import ase.io
import torch
import torch_geometric
import yaml
from torch.utils.data import DataLoader

from ocpmodels.common.ase_utils import OCPCalculator, Relaxation, relax_eval
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import plot_histogram, save_checkpoint
from ocpmodels.datasets import (
    TrajectoryDataset,
    TrajectoryLmdbDataset,
    data_list_collater,
)
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

        if isinstance(dataset, list):
            self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
        else:
            self.config["dataset"] = dataset

        if not is_debug:
            os.makedirs(self.config["cmd"]["checkpoint_dir"])
            os.makedirs(self.config["cmd"]["results_dir"])
            os.makedirs(self.config["cmd"]["logs_dir"])

        self.is_debug = is_debug
        self.is_vis = is_vis
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        if self.config["task"]["dataset"] == "trajectory_lmdb":
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config["optim"]["batch_size"],
                shuffle=True,
                collate_fn=data_list_collater,
                num_workers=self.config["optim"]["num_workers"],
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
                    collate_fn=data_list_collater,
                    num_workers=self.config["optim"]["num_workers"],
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
                batch_size=self.config["optim"]["batch_size"]
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
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                if "target_mean" in self.config["dataset"]:
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

        if self.is_vis and self.config["task"]["dataset"] != "qm9":
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
    def predict(self, dataset, batch_size=32):
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
                collate_fn=data_list_collater,
            )
        elif isinstance(dataset, torch_geometric.data.Batch):
            data_loader = [dataset]
        else:
            raise NotImplementedError

        self.model.eval()
        predictions = {"energy": [], "forces": []}

        for i, batch in enumerate(data_loader):
            batch.to(self.device)
            out, _ = self._forward(batch, compute_metrics=False)
            if self.normalizers is not None and "target" in self.normalizers:
                out["output"] = self.normalizers["target"].denorm(
                    out["output"]
                )
                out["force_output"] = self.normalizers["grad_target"].denorm(
                    out["force_output"]
                )
            atoms_sum = 0
            predictions["energy"].extend(out["output"].tolist())
            for natoms in batch.natoms:
                predictions["forces"].append(
                    out["force_output"][atoms_sum : natoms + atoms_sum]
                    .cpu()
                    .detach()
                    .numpy()
                )
                atoms_sum += natoms

        return predictions

    def train(self):
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)

                # Forward, loss, backward.
                out, metrics = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Update meter.
                meter_update_dict = {
                    "epoch": epoch + (i + 1) / len(self.train_loader),
                    "loss": loss.item(),
                }
                meter_update_dict.update(metrics)
                self.meter.update(meter_update_dict)

                # Make plots.
                if self.logger is not None:
                    self.logger.log(
                        meter_update_dict,
                        step=epoch * len(self.train_loader) + i + 1,
                        split="train",
                    )

                # Print metrics.
                if i % self.config["cmd"]["print_every"] == 0:
                    print(self.meter)

            self.scheduler.step()

            if self.val_loader is not None:
                self.validate(split="val", epoch=epoch)

            if self.test_loader is not None:
                self.validate(split="test", epoch=epoch)

            if (
                "relaxation_dir" in self.config["task"]
                and self.config["task"].get("ml_relax", "end") == "train"
            ):
                self.validate_relaxation(
                    split="val", epoch=epoch,
                )

            if not self.is_debug:
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
                    },
                    self.config["cmd"]["checkpoint_dir"],
                )
        if (
            "relaxation_dir" in self.config["task"]
            and self.config["task"].get("ml_relax", "end") == "end"
        ):
            self.validate_relaxation(
                split="val", epoch=epoch,
            )

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))
        self.model.eval()

        meter = Meter(split=split)

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in enumerate(loader):
            batch = batch.to(self.device)

            # Forward.
            out, metrics = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Update meter.
            meter_update_dict = {"loss": loss.item()}
            meter_update_dict.update(metrics)
            meter.update(meter_update_dict)

        # Make plots.
        if self.logger is not None and epoch is not None:
            log_dict = meter.get_scalar_dict()
            log_dict.update({"epoch": epoch + 1})
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        print(meter)

    def validate_relaxation(self, split="val", epoch=None):
        print("### Evaluating ML-relaxation")
        self.model.eval()
        metrics = {}
        meter = Meter(split=split)

        mae_energy, mae_structure = relax_eval(
            trainer=self,
            traj_dir=self.config["task"]["relaxation_dir"],
            metric=self.config["task"]["metric"],
            steps=self.config["task"].get("relaxation_steps", 300),
            fmax=self.config["task"].get("relaxation_fmax", 0.01),
            results_dir=self.config["cmd"]["results_dir"],
        )

        metrics[
            "relaxed_energy/{}".format(self.config["task"]["metric"])
        ] = mae_energy

        metrics[
            "relaxed_structure/{}".format(self.config["task"]["metric"])
        ] = mae_structure

        meter.update(metrics)

        # Make plots.
        if self.logger is not None and epoch is not None:
            log_dict = meter.get_scalar_dict()
            log_dict.update({"epoch": epoch + 1})
            self.logger.log(
                log_dict,
                step=(epoch + 1) * len(self.train_loader),
                split=split,
            )

        print(meter)

        return mae_energy, mae_structure
