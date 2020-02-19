import os
import torch
from torch_geometric.data import DataLoader, InMemoryDataset


class UlissiDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transfrom=None):
        super(UlissiDataset, self).__init__(root, transform, pre_transform=None)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load('./data_H_4k_surface.pt')

    @property
    def processed_file_names(self):

        return "data.pt"

    def _download(self):
        pass

    def _process(self):
        pass


def get_data_loaders(save_dir, batch_size):

    train_dataset = torch.load(os.path.join(save_dir, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(save_dir, "val_dataset.pt"))
    test_dataset = torch.load(os.path.join(save_dir, "test_dataset.pt"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
