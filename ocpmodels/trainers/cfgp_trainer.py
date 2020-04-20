import torch

from ..common.registry import registry
from ..modules.normalizer import Normalizer
from ..trainers.simple_trainer import SimpleTrainer
from ..trainers.gpytorch_trainer import GPyTorchTrainer


@registry.register_trainer("cfgp")
class CfgpTrainer:
    def __init__(self, conv_trainer, gpytorch_trainer):
        '''
        The `conv_trainer` needs to be a `SimpleTrainer` whose model has the
        `_convolve` method. The `gpytorch_trainer` needs to be a
        `GPyTorchTrainer`.
        '''
        self.conv_trainer = conv_trainer
        self.gpytorch_trainer = gpytorch_trainer

        self.device = self.conv_trainer.device
        self.train_loader = self.conv_trainer.train_loader
        self.val_loader = self.conv_trainer.val_loader
        self.test_loader = self.conv_trainer.test_loader

    def train(self, lr=0.1, n_training_iter=20):
        print("### Beggining training on convolutional network.")
        self.conv_trainer.train()

        print("### Beggining training on GP.")
        convolutions = self._get_training_convolutions()
        train_indices = self.train_loader.dataset.__indices__
        train_y = self.train_loader.dataset.data.y[train_indices].to(self.device)
        self.gpytorch_trainer.train(train_x=convolutions,
                                    train_y=train_y,
                                    lr=lr,
                                    n_training_iter=n_training_iter)

    def _get_training_convolutions(self):
        train_convs = self._get_convolutions(self.train_loader)
        train_convs = torch.Tensor(train_convs).to(self.device)

        self.conv_normalizer = Normalizer(train_convs, self.device)
        normed_convs = self.conv_normalizer.norm(train_convs)
        return normed_convs

    def _get_convolutions(self, data_loader):
        self.conv_trainer.model.eval()
        convolutions = []

        for i, batch in enumerate(data_loader):
            batch.to(self.device)
            out = self.conv_trainer.model._convolve(batch)
            for conv in out.tolist():
                convolutions.append(conv)

        return convolutions

    def predict(self, src, batch_size=32):
        print("### Generating predictions on {}.".format(src))

        # Parse the data
        dataset_config = {"src": src}
        dataset = registry.get_dataset_class(self.conv_trainer.config["task"]["dataset"])(
            dataset_config
        )
        data_loader = dataset.get_full_dataloader(batch_size=batch_size)

        # Get the convolutions
        convs = self._get_convolutions(data_loader)
        convs = torch.Tensor(convs).to(self.device)
        try:
            normed_convs = self.conv_normalizer.norm(convs)
        except AttributeError as error:
            raise type(error)(error.message + '; error may have occurred '
                              'because the CFGP may not have been trained yet')

        # Feed the convolutions into the GP
        targets_pred, targets_std = self.gpytorch_trainer.predict(normed_convs)
        return targets_pred, targets_std
