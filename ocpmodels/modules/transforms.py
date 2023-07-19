import torch

from ocpmodels.common.utils import cg_decomp_mat, irreps_sum


class DataTransforms:
    def __init__(self, config):
        self.config = config

    def __call__(self, data_object):
        if self.config is None:
            return data_object

        for transform_fn in self.config:
            data_object = eval(transform_fn)(
                data_object, self.config[transform_fn]
            )

        return data_object


def decompose_tensor(data_object, config):
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
