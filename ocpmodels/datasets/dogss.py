from ocpmodels.common.registry import registry
from ocpmodels.datasets import BaseDataset
from torch_geometric.data import DataLoader, InMemoryDataset

@registry.register_dataset("dogss")
class DOGSS(BaseDataset):
    
    def __init__(self, config, transform=None, pre_transform=None):
        super(DOGSS, self).__init__(config, transform, pre_transform)
        
    def get_dataloaders(self, batch_size=None):
        assert batch_size is not None
        assert self.train_size + self.val_size + self.test_size <= len(self)
        
        self = self.shuffle()
        
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
