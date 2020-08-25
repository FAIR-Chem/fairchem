import datetime
import os

import torch
import yaml
from torch.utils.data import DataLoader

from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.meter import Meter, mae
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import save_checkpoint
from ocpmodels.datasets import SinglePointLmdbDataset, data_list_collater
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("energy")
class EnergyTrainer(BaseTrainer):
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
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            self.output_device = self.config["optim"].get(
                "output_device", device_ids[0]
            )
            self.device = f"cuda:{self.output_device}"
        else:
            self.device = "cpu"

        print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))

        self.parallel_collater = ParallelCollater(
            self.config["optim"].get("num_gpus", 1)
        )
        if self.config["task"]["dataset"] == "single_point_lmdb":
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config["optim"]["batch_size"],
                shuffle=True,
                collate_fn=self.parallel_collater,
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
                    collate_fn=self.parallel_collater,
                    num_workers=self.config["optim"]["num_workers"],
                )
        else:
            raise NotImplementedError

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
                raise NotImplementedError

    def load_model(self):
        super(EnergyTrainer, self).load_model()

        self.model = OCPDataParallel(
            self.model,
            output_device=self.output_device,
            num_gpus=self.config["optim"].get("num_gpus", 1),
        )
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
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
            torch.cuda.empty_cache()

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

    def validate(self, split="val", epoch=None):
        print("### Evaluating on {}.".format(split))
        self.model.eval()

        meter = Meter(split=split)

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in enumerate(loader):
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

    def _forward(self, batch_list, compute_metrics=True):
        out = {}

        # forward pass.
        output = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        out["output"] = output

        if not compute_metrics:
            return out, None

        metrics = {}
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.config["dataset"].get("normalize_labels", True):
            errors = eval(self.config["task"]["metric"])(
                self.normalizers["target"].denorm(output).cpu(),
                energy_target.cpu(),
            ).view(-1)
        else:
            errors = eval(self.config["task"]["metric"])(
                output.cpu(), energy_target.cpu()
            ).view(-1)

        for i, label in enumerate(self.config["task"]["labels"]):
            metrics[
                "{}/{}".format(label, self.config["task"]["metric"])
            ] = errors[i]

        return out, metrics

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.config["dataset"].get("normalize_labels", True):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.criterion(out["output"], target_normed)
        return loss
