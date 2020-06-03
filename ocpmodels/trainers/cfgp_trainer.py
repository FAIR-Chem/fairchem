import pickle
import torch

from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer


@registry.register_trainer("cfgp")
class CfgpTrainer:
    def __init__(self, conv_trainer, gpytorch_trainer):
        """
        The `conv_trainer` needs to be a `SimpleTrainer` whose model has the
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
        print("### Beginning training on convolutional network.")
        self.conv_trainer.train()
        self._train_gp(lr, n_training_iter)

    def _train_gp(self, lr, n_training_iter):
        print("### Beginning training on GP.")
        convolutions, train_y = self._get_training_convolutions()
        self.gpytorch_trainer.train(
            train_x=convolutions,
            train_y=train_y,
            lr=lr,
            n_training_iter=n_training_iter,
        )

    def _get_training_convolutions(self):
        train_convs, train_y = self._get_convolutions(self.train_loader)

        self.conv_normalizer = Normalizer(train_convs, self.device)
        normed_convs = self.conv_normalizer.norm(train_convs)
        return normed_convs, train_y

    def _get_convolutions(self, data_loader):
        self.conv_trainer.model.eval()
        convolutions = []
        targets = []

        for i, batch in enumerate(data_loader):
            batch.to(self.device)
            out = self.conv_trainer.model._convolve(batch)
            for conv, target in zip(out.tolist(), batch.y):
                convolutions.append(conv)
                targets.append(target)

        convolutions = torch.Tensor(convolutions).to(self.device)
        targets = torch.Tensor(targets).to(self.device)
        return convolutions, targets

    def predict(self, src, batch_size=32):
        print("### Generating predictions on {}.".format(src))

        # Parse the data
        dataset_config = {"src": src}
        dataset = registry.get_dataset_class(
            self.conv_trainer.config["task"]["dataset"]
        )(dataset_config)
        data_loader = dataset.get_full_dataloader(batch_size=batch_size)

        # Get the convolutions
        convs, targets_actual = self._get_convolutions(data_loader)
        try:
            normed_convs = self.conv_normalizer.norm(convs)
        except AttributeError as error:
            raise type(error)(str(error) + "; error may have occurred "
                              "because the CFGP may not have been trained yet")

        # Feed the convolutions into the GP
        targets_pred, targets_std = self.gpytorch_trainer.predict(normed_convs)
        return targets_pred, targets_std

    def save_state(self, gp_path='gp_state.pth', normalizer_path='normalizer.pth'):
        self.gpytorch_trainer.save_state(gp_path)
        with open(normalizer_path, 'wb') as f:
            pickle.dump(self.conv_normalizer.state_dict(), f)

    def load_state(self, nn_checkpoint_file, gp_checkpoint_file,
                   normalizer_checkpoint_file):
        self._load_conv(nn_checkpoint_file)
        self._load_gp(gp_checkpoint_file)
        self._load_normalizer(normalizer_checkpoint_file)

    def _load_conv(self, nn_checkpoint_file):
        self.conv_trainer.load_state(nn_checkpoint_file)

    def _load_gp(self, gp_checkpoint_file):
        convolutions, train_y = self._get_training_convolutions()
        self.gpytorch_trainer.load_state(gp_checkpoint_file, convolutions, train_y)

    def _load_normalizer(self, normalizer_checkpoint_file):
        with open(normalizer_checkpoint_file, 'rb') as f:
            normalizer_state_dict = pickle.load(f)
        self.conv_normalizer.load_state_dict(normalizer_state_dict)
