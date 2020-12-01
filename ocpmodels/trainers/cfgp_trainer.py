import pickle

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer


@registry.register_trainer("cfgp")
class CfgpTrainer:
    def __init__(self, conv_trainer, gpytorch_trainer):
        """
        The `conv_trainer` needs to be a `EnergyTrainer` whose model has the
        `_convolve` method. The `gpytorch_trainer` needs to be a
        `GPyTorchTrainer`.
        """
        self.conv_trainer = conv_trainer
        self.gpytorch_trainer = gpytorch_trainer

        self.device = self.conv_trainer.device
        self.train_loader = self.conv_trainer.train_loader
        self.val_loader = self.conv_trainer.val_loader
        self.test_loader = self.conv_trainer.test_loader

    def train(self, lr=0.1, n_training_iter=20):
        if distutils.is_master():
            print("### Beginning training on convolutional network.")
        self.conv_trainer.train()
        self._train_gp(lr, n_training_iter)
        distutils.synchronize()

    def _train_gp(self, lr, n_training_iter):
        convolutions, train_y = self._get_training_convolutions()
        if distutils.is_master():
            self.gpytorch_trainer.train(
                train_x=convolutions,
                train_y=train_y,
                lr=lr,
                n_training_iter=n_training_iter,
            )

    def _get_training_convolutions(self):
        train_convs, train_y = self._get_convolutions(self.train_loader)
        if distutils.initialized():
            train_convs = torch.cat(
                distutils.all_gather(train_convs, device=self.device), dim=0
            )
            train_y = torch.cat(
                distutils.all_gather(train_y, device=self.device)
            )

        self.conv_normalizer = Normalizer(train_convs, device=self.device)
        normed_convs = self.conv_normalizer.norm(train_convs)
        return normed_convs, train_y

    def _get_test_convolutions(self, data_loader):
        convs, targets = self._get_convolutions(data_loader)
        if distutils.initialized():
            convs = torch.cat(
                distutils.all_gather(convs, device=self.device), dim=0
            )
            targets = torch.cat(
                distutils.all_gather(targets, device=self.device)
            )

        try:
            normed_convs = self.conv_normalizer.norm(convs)
        except AttributeError as error:
            raise type(error)(
                str(error) + "; error may have occurred "
                "because the CFGP may not have been trained yet"
            )
        return normed_convs, targets

    def _get_convolutions(self, data_loader):
        self.conv_trainer.model.eval()
        module = self.conv_trainer.model.module
        # DDP models are wrapped in DistributedDataParallel
        if distutils.initialized():
            module = module.module
        convolutions = []
        targets = []

        for batches in data_loader:
            for batch in batches:
                out = module._convolve(batch.to(self.device))
                for conv, target in zip(out.tolist(), batch.y_relaxed):
                    convolutions.append(conv)
                    targets.append(target)

        convolutions = torch.Tensor(convolutions).to(self.device)
        targets = torch.Tensor(targets).to(self.device)
        return convolutions, targets

    def predict(self, src, batch_size=32):
        if distutils.is_master():
            print(f"### Generating predictions on {src}")

        # Parse the data
        dataset_config = {"src": src}
        dataset = registry.get_dataset_class(
            self.conv_trainer.config["task"]["dataset"]
        )(dataset_config)
        test_sampler = DistributedSampler(
            dataset,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            shuffle=False,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.conv_trainer.parallel_collater,
            num_workers=self.conv_trainer.config["optim"]["num_workers"],
            sampler=test_sampler,
        )

        # Get the convolutions
        normed_convs, targets = self._get_test_convolutions(data_loader)

        # Feed the convolutions into the GP
        if distutils.is_master():
            targets_pred, targets_std = self.gpytorch_trainer.predict(
                normed_convs
            )
            results = {"pred": targets_pred, "std": targets_std}
            results_path = f'{self.conv_trainer.config["cmd"]["results_dir"]}/predictions.pt'
            torch.save(results, results_path)
            print(f"### Predictions saved to {results_path}")

    def save_state(
        self, gp_path="gp_state.pth", normalizer_path="normalizer.pth"
    ):
        self.gpytorch_trainer.save_state(gp_path)
        with open(normalizer_path, "wb") as f:
            pickle.dump(self.conv_normalizer.state_dict(), f)

    def load_state(
        self,
        nn_checkpoint_file,
        gp_checkpoint_file,
        normalizer_checkpoint_file,
    ):
        self._load_conv(nn_checkpoint_file)
        self._load_gp(gp_checkpoint_file)
        self._load_normalizer(normalizer_checkpoint_file)

    def _load_conv(self, nn_checkpoint_file):
        self.conv_trainer.load_pretrained(nn_checkpoint_file)

    def _load_gp(self, gp_checkpoint_file):
        convolutions, train_y = self._get_training_convolutions()
        self.gpytorch_trainer.load_state(
            gp_checkpoint_file, convolutions, train_y
        )

    def _load_normalizer(self, normalizer_checkpoint_file):
        with open(normalizer_checkpoint_file, "rb") as f:
            normalizer_state_dict = pickle.load(f)
        self.conv_normalizer.load_state_dict(normalizer_state_dict)
