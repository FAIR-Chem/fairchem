from typing import Callable, Dict, List

import torch
from torch_geometric.data import Data

from ocpmodels.common.utils import cg_decomp_mat, irreps_sum
from ocpmodels.modules.normalizer import normalizer_transform


class DataTransforms:
    def __init__(self, transform_config: List[Dict]) -> None:
        transform_config = transform_config.copy()

        self.transforms: List[Callable] = []
        for transform_dict in transform_config:
            for name, config in transform_dict.items():
                if name == "normalizer":
                    fn = normalizer_transform(config)
                else:
                    fn = eval(name)

                self.transforms.append(fn)

    def __call__(self, data_object):
        for transform in self.transforms:
            data_object = transform(data_object)
        return data_object


def decompose_tensor(data_object, config) -> Data:
    tensor_key = config["tensor"]
    rank = config["rank"]

    if rank != 2:
        raise NotImplementedError

    tensor_decomposition = torch.einsum(
        "ab, cb->ca",
        cg_decomp_mat(rank),
        data_object[tensor_key].reshape(1, irreps_sum(rank)),
    )

    for decomposition_key in config["decomposition"]:
        irrep_dim = config["decomposition"][decomposition_key]["irrep_dim"]
        data_object[decomposition_key] = tensor_decomposition[
            :,
            max(0, irreps_sum(irrep_dim - 1)) : irreps_sum(irrep_dim),
        ]

    return data_object


def flatten(data_object, config) -> Data:
    tensor_key = config["tensor"]

    data_object[tensor_key] = data_object[tensor_key].reshape(1, -1)

    return data_object
