from torch_geometric.data import DataLoader

from baselines.common.registry import registry
from baselines.datasets.base import BaseDataset


@registry.register_dataset("iso17")
class ISO17(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(ISO17, self).__init__(config, transform, pre_transform)

    @property
    def test_size(self):
        if self.config["test_fold"] == "test_within":
            return 101000
        elif self.config["test_fold"] == "test_other":
            return 130000

    def get_dataloaders(self, batch_size=None):
        assert batch_size is not None
        assert self.train_size + self.val_size + 101000 + 130000 <= len(self)

        if self.config["test_fold"] == "test_within":
            test_dataset = self[-(101000 + 130000) : -130000]
        elif self.config["test_fold"] == "test_other":
            test_dataset = self[-130000:]

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
