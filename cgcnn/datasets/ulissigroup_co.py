from cgcnn.common.registry import registry
from cgcnn.datasets.base import BaseDataset


@registry.register_dataset("ulissigroup_co")
class UlissigroupCO(BaseDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(UlissigroupCO, self).__init__(root, transform, pre_transform)
