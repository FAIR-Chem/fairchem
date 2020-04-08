import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9

from baselines.common.registry import registry
from baselines.common.utils import Complete
from baselines.datasets.base import BaseDataset


@registry.register_dataset("qm9")
class QM9Dataset(QM9):
    def __init__(self, config):
        transform = T.Compose([Complete(), T.Distance(norm=False)])
        self.config = config
        return super(QM9Dataset, self).__init__(
            config["src"], transform=transform
        )

    @property
    def train_size(self):
        return self.config["train_size"]

    @property
    def val_size(self):
        return self.config["val_size"]

    @property
    def test_size(self):
        return self.config["test_size"]

    def get_dataloaders(self, batch_size=None):
        assert batch_size is not None
        assert self.train_size + self.val_size + self.test_size <= len(self)

        test_dataset = self[-self.test_size :]
        train_val_dataset = self[: self.train_size + self.val_size].shuffle()

        train_loader = DataLoader(
            train_val_dataset[: self.train_size],
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            train_val_dataset[
                self.train_size : self.train_size + self.val_size
            ],
            batch_size=batch_size,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader
