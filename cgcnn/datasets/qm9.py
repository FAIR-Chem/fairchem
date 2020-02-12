import torch_geometric.transforms as T
from torch_geometric.datasets import QM9

from cgcnn.common.registry import registry
from cgcnn.common.utils import Complete
from cgcnn.datasets.base import BaseDataset


@registry.register_dataset("qm9")
class QM9Dataset(QM9):
    def __init__(self, root):
        transform = T.Compose([Complete(), T.Distance(norm=False)])
        return super(QM9Dataset, self).__init__(root, transform=transform)
