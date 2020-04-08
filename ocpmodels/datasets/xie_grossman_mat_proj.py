from ocpmodels.common.registry import registry
from ocpmodels.datasets.base import BaseDataset


@registry.register_dataset("xie_grossman_mat_proj")
class XieGrossmanMatProj(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(XieGrossmanMatProj, self).__init__(
            config, transform, pre_transform
        )
