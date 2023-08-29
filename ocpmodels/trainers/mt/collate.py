"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


from logging import getLogger

from torch_geometric.data import Batch, Data

from .config import DatasetConfig

log = getLogger(__name__)


def data_list_collater(
    data_list: list[Data],
    dataset_config: DatasetConfig | None = None,
    otf_graph: bool = False,
):
    batch = Batch.from_data_list(
        data_list,
        exclude_keys=dataset_config.collate_exclude_keys
        if dataset_config is not None
        else None,
    )

    if not otf_graph:
        raise NotImplementedError("OTF graph is mandatory for MT trainer.")

    return batch


class ParallelCollater:
    def __init__(
        self,
        num_gpus: int,
        dataset_config: DatasetConfig | None = None,
        otf_graph: bool = False,
    ) -> None:
        if not otf_graph:
            raise NotImplementedError("OTF graph is mandatory for MT trainer.")

        self.num_gpus = num_gpus
        self.dataset_config = dataset_config
        self.otf_graph = otf_graph

    def __call__(self, data_list):
        if self.num_gpus not in (0, 1):  # 0=cpu
            raise NotImplementedError("DP not supported for MT. Use DDP.")

        batch = data_list_collater(
            data_list,
            self.dataset_config,
            otf_graph=self.otf_graph,
        )
        return [batch]
