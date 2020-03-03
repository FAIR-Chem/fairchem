import numpy as np

import torch
from baselines.common.registry import registry
from baselines.datasets.base import BaseDataset
from torch_geometric.data import DataLoader


@registry.register_dataset("active_discovery")
class ActiveDiscoveryDataset(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(ActiveDiscoveryDataset, self).__init__(
            config, transform, pre_transform
        )

        # TODO: load training data indices here. It should be a torch.LongTensor
        # assert 0 == 1
        train_indices = np.arange(self.config["max_train_size"]).tolist()
        self.train_index_groups = np.array_split(train_indices, 14)
        self.train_group_size = 1
        # TODO ends here.

    @property
    def train_size(self):
        return sum(
            [len(x) for x in self.train_index_groups[: self.train_group_size]]
        )

    @property
    def init_train_size(self):
        return self.config["init_train_size"]

    @property
    def max_train_size(self):
        return self.config["max_train_size"]

    def increase_training_data(self):
        # TODO: How do you want to augment the training set. Specify here
        # or in init.
        # assert 0 == 1
        if self.train_size < self.max_train_size:
            self.train_group_size += 1
        # TODO ends here.

    def get_train_dataloader(self, batch_size=None):
        assert self.train_size + self.val_size <= len(self)
        train_indices = self.train_index_groups[0].tolist()
        for _ in range(1, self.train_group_size):
            train_indices.extend(self.train_index_groups[_].tolist())
        train_dataset = self[torch.LongTensor(train_indices)].shuffle()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        return train_loader

    def get_val_dataloader(self, batch_size=None):
        val_dataset = self[-self.val_size :]
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return val_loader

    def get_dataloaders(self, batch_size=None):
        raise NotImplementedError
