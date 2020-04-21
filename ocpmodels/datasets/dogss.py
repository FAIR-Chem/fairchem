"""
Note that much of this code was taken from
`https://github.com/ulissigroup/cgcnn/`, which in turn was based on
`https://github.com/txie-93/cgcnn`.
"""

import os
from itertools import product

import ase.db
import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data, InMemoryDataset, DataLoader

from ..common.registry import registry
from .base import BaseDataset
from .elemental_embeddings import EMBEDDINGS



@registry.register_dataset("DOGSS")
class DOGSS(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(BaseDataset, self).__init__(config["src"], transform=None, pre_transform=None)

        self.data, self.slices = torch.load("ocpmodels/datasets/data_surfaces.pt")
        self.config = config
        
    @property
    def processed_file_names(self):
        return "data_surfaces.pt"
