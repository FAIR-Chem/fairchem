from ocpmodels.common.registry import registry
from ocpmodels.datasets import BaseDataset


@registry.register_dataset("dogss")
class DOGSS(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(DOGSS, self).__init__(config, transform, pre_transform)
