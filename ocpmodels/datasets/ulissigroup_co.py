from ocpmodels.common.registry import registry
from ocpmodels.datasets.base import BaseDataset


@registry.register_dataset("ulissigroup_co")
class UlissigroupCO(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(UlissigroupCO, self).__init__(config, transform, pre_transform)
