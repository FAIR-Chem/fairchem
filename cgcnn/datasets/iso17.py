from cgcnn.common.registry import registry
from cgcnn.datasets.base import BaseDataset


@registry.register_dataset("iso17")
class ISO17(BaseDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ISO17, self).__init__(root, transform, pre_transform)
