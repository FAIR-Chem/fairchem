import torch
from baselines.common.registry import registry
from baselines.common.utils import plot_histogram, save_checkpoint
from baselines.modules.normalizer import Normalizer
from baselines.trainers import BaseTrainer


@registry.register_trainer("active_discovery")
class ActiveDiscoveryTrainer(BaseTrainer):
    def __init__(self, args):
        super(ActiveDiscoveryTrainer, self).__init__(args)

    def load_task(self):
        print("### Loading dataset: {}".format(self.config["task"]["dataset"]))
        self.dataset = registry.get_dataset_class(
            self.config["task"]["dataset"]
        )(self.config["dataset"])
        num_targets = 1

        self.num_targets = num_targets
        self.train_loader = self.dataset.get_train_dataloader(
            batch_size=self.config["optim"]["batch_size"]
        )
        self.val_loader = self.dataset.get_val_dataloader(
            batch_size=self.config["optim"]["batch_size"]
        )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        self.normalizers["target"] = Normalizer(
            self.train_loader.dataset.data.y, self.device
        )

        if self.is_vis:
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
            ]
            self.logger.log_plots(plots)

    def train(self):
        # TODO(abhshkdz): Timers for dataloading and forward pass.
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.model.train()

            # TODO: when / how should we increase the training data size.
            if (
                epoch != 0
                and epoch % self.config["dataset"]["increase_data_every"] == 0
            ):
                self.dataset.increase_training_data()
                self.train_loader = self.dataset.get_train_dataloader(
                    batch_size=self.config["optim"]["batch_size"]
                )

            print(
                "Epoch: {}, training data size: {}".format(
                    epoch, len(self.train_loader.dataset)
                )
            )

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

            with torch.no_grad():
                self.validate(split="val", epoch=epoch)

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
