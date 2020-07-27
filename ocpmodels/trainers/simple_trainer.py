import datetime
import os
import warnings

import torch
import yaml

from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.meter import Meter, mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import plot_histogram, save_checkpoint
from ocpmodels.datasets import *
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("simple")
class SimpleTrainer(BaseTrainer):
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
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            self.output_device = self.config["optim"].get(
                "output_device", device_ids[0]
            )
            self.device = f"cuda:{self.output_device}"
        else:
            self.device = "cpu"

        self.parallel_collater = ParallelCollater(
            self.config["optim"].get("num_gpus", 1)
        )
        print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    def load_model(self):
        super(SimpleTrainer, self).load_model()

        self.model = OCPDataParallel(
            self.model,
            output_device=self.output_device,
            num_gpus=self.config["optim"].get("num_gpus", 1),
        )
        self.model.to(self.device)

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            self.config["dataset"]
        )

        if self.config["task"]["dataset"] in ["qm9", "dogss"]:
            num_targets = dataset.data.y.shape[-1]
            if (
                "label_index" in self.config["task"]
                and self.config["task"]["label_index"] is not False
            ):
                dataset.data.y = dataset.data.y[
                    :, int(self.config["task"]["label_index"])
                ]
                num_targets = 1
        else:
            num_targets = 1

        self.num_targets = num_targets
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = dataset.get_dataloaders(
            batch_size=int(self.config["optim"]["batch_size"]),
            collate_fn=self.parallel_collater,
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

    def train(self, max_epochs=None, return_metrics=False):
        # TODO(abhshkdz): Timers for dataloading and forward pass.
        num_epochs = (
            max_epochs
            if max_epochs is not None
            else self.config["optim"]["max_epochs"]
        )
        for epoch in range(num_epochs):
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

            with torch.no_grad():
                if self.val_loader is not None:
                    v_loss, v_mae = self.validate(split="val", epoch=epoch)

                if self.test_loader is not None:
                    test_loss, test_mae = self.validate(
                        split="test", epoch=epoch
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
        if return_metrics:
            return {
                "training_loss": float(self.meter.loss.global_avg),
                "training_mae": float(
                    self.meter.meters[
                        self.config["task"]["labels"][0]
                        + "/"
                        + self.config["task"]["metric"]
                    ].global_avg
                ),
                "validation_loss": v_loss,
                "validation_mae": v_mae,
                "test_loss": test_loss,
                "test_mae": test_mae,
            }

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
        return (
            float(meter.loss.global_avg),
            float(
                meter.meters[
                    self.config["task"]["labels"][0]
                    + "/"
                    + self.config["task"]["metric"]
                ].global_avg
            ),
        )

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

        if "grad_input" in self.config["task"]:
            force_pred = force_output
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )

            if self.config["task"].get("eval_on_free_atoms", True):
                fixed = torch.cat([batch.fixed for batch in batch_list])
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

        if self.config["dataset"].get("normalize_labels", True):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss.append(self.criterion(out["output"], target_normed))

        # TODO(abhshkdz): Test support for gradients wrt input.
        # TODO(abhshkdz): Make this general; remove dependence on `.forces`.
        if "grad_input" in self.config["task"]:
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.config["dataset"].get("normalize_labels", True):
                grad_target_normed = self.normalizers["grad_target"].norm(
                    force_target
                )
            else:
                grad_target_normed = force_target

            # Force coefficient = 30 has been working well for us.
            force_mult = self.config["optim"].get("force_coefficient", 30)
            if self.config["task"].get("train_on_free_atoms", False):
                fixed = torch.cat([batch.fixed for batch in batch_list])
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

    # Takes in a new data source and generates predictions on it.
    def predict(self, src, batch_size=32):
        print("### Generating predictions on {}.".format(src))

        dataset_config = {"src": src}
        dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
            dataset_config
        )
        data_loader = dataset.get_full_dataloader(
            batch_size=batch_size, collate_fn=self.parallel_collater
        )

        self.model.eval()
        predictions = []

        for i, batch in enumerate(data_loader):
            out, metrics = self._forward(batch)
            if self.config["dataset"].get("normalize_labels", True):
                out["output"] = self.normalizers["target"].denorm(
                    out["output"]
                )
            predictions.extend(out["output"].tolist())

        return predictions

    def load_state(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file)["state_dict"]
        self.model.load_state_dict(state_dict)
