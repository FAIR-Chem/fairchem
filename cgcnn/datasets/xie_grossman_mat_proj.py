from cgcnn.common.registry import registry
from cgcnn.datasets.base import BaseDataset


@registry.register_dataset("xie_grossman_mat_proj")
class XieGrossmanMatProj(BaseDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(XieGrossmanMatProj, self).__init__(
            root, transform, pre_transform
        )
