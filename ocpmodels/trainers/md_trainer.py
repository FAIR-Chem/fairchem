import datetime
import os
import warnings

import torch
import yaml
from ocpmodels.common.meter import Meter
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    add_edge_distance_to_graph,
    plot_histogram,
    save_checkpoint,
)
from ocpmodels.datasets import *
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("md")
class MDTrainer(BaseTrainer):
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
        logger="wandb",
    ):

        if run_dir is None:
            run_dir = os.getcwd()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if identifier:
            timestamp += "-{}".format(identifier)

        self.config = {
            "task": task,
            "dataset": dataset,
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
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            self.config["dataset"]
        )
        num_targets = 1

        self.num_targets = num_targets
        self.train_loader, self.val_loader, self.test_loader = dataset.get_dataloaders(
            batch_size=self.config["optim"]["batch_size"]
        )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config["dataset"].get("normalize_labels", True):
            self.normalizers["target"] = Normalizer(
                self.train_loader.dataset.data.y[
                    self.train_loader.dataset.__indices__
                ],
                self.device,
            )

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if "grad_input" in self.config["task"]:
            if self.config["dataset"].get("normalize_labels", True):
                self.normalizers["grad_target"] = Normalizer(
                    self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    self.device,
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
    def predict(self, dataset_config, batch_size=32):
        print(
            "### Generating predictions on {}.".format(dataset_config["src"])
        )

        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            dataset_config
        )
        data_loader = dataset.get_full_dataloader(batch_size=batch_size)

        self.model.eval()
        predictions = []

        for i, batch in enumerate(data_loader):
            batch.to(self.device)
            out, metrics = self._forward(batch)
            if self.normalizers is not None and "target" in self.normalizers:
                out["output"] = self.normalizers["target"].denorm(
                    out["output"]
                )
            predictions.extend(out["output"].tolist())

        return predictions

    def train(self):
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                batch = add_edge_distance_to_graph(batch, device=self.device)

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

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))
        self.model.eval()

        meter = Meter(split=split)

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            batch = add_edge_distance_to_graph(batch, device=self.device)

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
