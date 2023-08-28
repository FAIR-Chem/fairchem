import torch
from torch_geometric.data import Data


def cg_decomp_mat(l: int, device: torch.device | None = None):
    if l not in (2,):
        raise NotImplementedError

    change_mat = torch.tensor(
        [
            [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
            [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
            [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
            [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
            [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
            [
                -(6 ** (-0.5)),
                0,
                0,
                0,
                2 * 6 ** (-0.5),
                0,
                0,
                0,
                -(6 ** (-0.5)),
            ],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
            [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
        ],
        device=device,
    ).detach()

    return change_mat


def irreps_sum(l: int):
    total = 0
    for i in range(l + 1):
        total += 2 * i + 1

    return total


class DataTransforms:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, data_object):
        if not self.config:
            return data_object

        for transform_fn in self.config:
            # TODO move normalizer into dataset
            if transform_fn == "normalizer":
                continue
            data_object = eval(transform_fn)(data_object, self.config[transform_fn])

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
