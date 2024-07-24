from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fairchem.core.common.utils import cg_change_mat, irreps_sum

if TYPE_CHECKING:
    from torch_geometric.data import Data


class DataTransforms:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, data_object):
        if not self.config:
            return data_object

        for transform_fn in self.config:
            # TODO: Normalization information used in the trainers. Ignore here
            # for now.
            if transform_fn == "normalizer":
                continue
            data_object = eval(transform_fn)(data_object, self.config[transform_fn])

        return data_object
    
def shift_energy(data_object, config) -> Data:
    data_object.energy = data_object.energy - config
    return data_object

def add_group(data_object, config) -> Data:
    data_object.group = config
    return data_object
    
def add_fixed_tensor(data_object, config) -> Data:
    data_object.fixed = torch.zeros((data_object.pos.size(0)), dtype=torch.float32, device=data_object.pos.device)
    return data_object

def decompose_tensor(data_object, config) -> Data:
    tensor_key = config["tensor"]
    rank = config["rank"]

    if tensor_key not in data_object:
        return data_object

    if rank != 2:
        raise NotImplementedError

    tensor_decomposition = torch.einsum(
        "ab, cb->ca",
        cg_change_mat(rank),
        data_object[tensor_key].reshape(1, irreps_sum(rank)),
    )

    for decomposition_key in config["decomposition"]:
        irrep_dim = config["decomposition"][decomposition_key]["irrep_dim"]
        data_object[decomposition_key] = tensor_decomposition[
            :,
            max(0, irreps_sum(irrep_dim - 1)) : irreps_sum(irrep_dim),
        ]

    return data_object
