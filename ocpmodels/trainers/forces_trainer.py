import datetime
import os

import ase.io
import torch
import torch_geometric
import yaml
from torch.utils.data import DataLoader
from torch_geometric.nn import DataParallel

from ocpmodels.common.ase_utils import OCPCalculator, Relaxation, relax_eval
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.meter import Meter, mae, mae_ratio, mean_l2_distance
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
        local_rank=0
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

    def load_model(self):
        super(ForcesTrainer, self).load_model()

        self.model = OCPDataParallel(
            self.model,
            output_device=self.output_device,
            num_gpus=self.config["optim"].get("num_gpus", 1),
        )
        self.model.to(self.device)

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
                collate_fn=self.parallel_collater,
            )
        elif isinstance(dataset, torch_geometric.data.Batch):
            data_loader = [[dataset]]
        else:
            raise NotImplementedError

        self.model.eval()
        predictions = {"energy": [], "forces": []}

        for i, batch_list in enumerate(data_loader):
            out, _ = self._forward(batch_list, compute_metrics=False)
            if self.normalizers is not None and "target" in self.normalizers:
                out["output"] = self.normalizers["target"].denorm(
                    out["output"]
                )
                out["force_output"] = self.normalizers["grad_target"].denorm(
                    out["force_output"]
                )
            atoms_sum = 0
            predictions["energy"].extend(out["output"].tolist())
            batch_natoms = torch.cat([batch.natoms for batch in batch_list])
            for natoms in batch_natoms:
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

    def _forward(self, batch_list, compute_metrics=True):
        out = {}

        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            output, output_forces = self.model(batch_list)
        else:
            output = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        out["output"] = output

        force_output = None
        if self.config["model_attributes"].get("regress_forces", True):
            out["force_output"] = output_forces
            force_output = output_forces

        if not compute_metrics:
            return out, None

        metrics = {}
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
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

        if (
            "label_index" in self.config["task"]
            and self.config["task"]["label_index"] is not False
        ):
            # TODO(abhshkdz): Get rid of this edge case for QM9.
            # This is only because QM9 has multiple targets and we can either
            # jointly predict all of them or one particular target.
            metrics[
                "{}/{}".format(
                    self.config["task"]["labels"][
                        self.config["task"]["label_index"]
                    ],
                    self.config["task"]["metric"],
                )
            ] = errors[0]
        else:
            for i, label in enumerate(self.config["task"]["labels"]):
                metrics[
                    "{}/{}".format(label, self.config["task"]["metric"])
                ] = errors[i]

        if self.config["model_attributes"].get("regress_forces", True):
            force_pred = force_output
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )

            if self.config["task"].get("eval_on_free_atoms", True):
                fixed = torch.cat(
                    [batch.fixed.to(self.device) for batch in batch_list]
                )
                mask = fixed == 0
                force_pred = force_pred[mask]
                force_target = force_target[mask]

            if self.config["dataset"].get("normalize_labels", True):
                grad_input_errors = eval(self.config["task"]["metric"])(
                    self.normalizers["grad_target"].denorm(force_pred).cpu(),
                    force_target.cpu(),
                )
            else:
                grad_input_errors = eval(self.config["task"]["metric"])(
                    force_pred.cpu(), force_target.cpu()
                )
            metrics[
                "force_x/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[0]
            metrics[
                "force_y/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[1]
            metrics[
                "force_z/{}".format(self.config["task"]["metric"])
            ] = grad_input_errors[2]

        return out, metrics

    def _compute_loss(self, out, batch_list):
        loss = []
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        force_target = torch.cat(
            [batch.force.to(self.device) for batch in batch_list], dim=0
        )

        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        if self.config["dataset"].get("normalize_labels", True):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss.append(energy_mult * self.criterion(out["output"], target_normed))

        # TODO(abhshkdz): Test support for gradients wrt input.
        # TODO(abhshkdz): Make this general; remove dependence on `.forces`.
        if self.config["model_attributes"].get("regress_forces", True):
            if self.config["dataset"].get("normalize_labels", True):
                grad_target_normed = self.normalizers["grad_target"].norm(
                    force_target
                )
            else:
                grad_target_normed = force_target

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
                        out["force_output"][mask], grad_target_normed[mask]
                    )
                )
            else:
                loss.append(
                    force_mult
                    * self.criterion(out["force_output"], grad_target_normed)
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss
