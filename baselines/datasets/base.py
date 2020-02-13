import torch
from torch_geometric.data import DataLoader, InMemoryDataset


class BaseDataset(InMemoryDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(BaseDataset, self).__init__(
            config["src"], transform, pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.config = config

    @property
    def processed_file_names(self):
        return "data.pt"

    def _download(self):
        pass

    def _process(self):
        pass

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
