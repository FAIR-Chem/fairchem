"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import heapq
import logging
from itertools import chain
from pathlib import Path
from typing import List, Literal, Protocol, Tuple, Union, runtime_checkable

import numba
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import BatchSampler, DistributedSampler, Sampler
from torch_geometric.data.data import BaseData

from ocpmodels.common import distutils, gp_utils
from ocpmodels.datasets import data_list_collater


class OCPDataParallel(torch.nn.DataParallel):
    use_cpu: bool

    def __init__(
        self, module, output_device: torch.device, num_gpus: int
    ) -> None:
        if num_gpus < 0:
            raise ValueError("# GPUs must be positive.")
        if num_gpus > torch.cuda.device_count():
            raise ValueError("# GPUs specified larger than available")

        self.src_device = torch.device(output_device)

        self.use_cpu = False
        if num_gpus == 0:
            self.use_cpu = True
        elif num_gpus == 1:
            device_ids = [self.src_device]
        else:
            if (
                self.src_device.type == "cuda"
                and self.src_device.index >= num_gpus
            ):
                raise ValueError("Main device must be less than # of GPUs")
            device_ids = list(range(num_gpus))

        if self.use_cpu:
            super(torch.nn.DataParallel, self).__init__()
            self.module = module

        else:
            super(OCPDataParallel, self).__init__(
                module=module,
                device_ids=device_ids,
                output_device=self.src_device,
            )

    def forward(self, batch_list, **kwargs):
        if self.use_cpu:
            return self.module(batch_list[0])

        if len(self.device_ids) == 1:
            return self.module(
                batch_list[0].to(f"cuda:{self.device_ids[0]}"), **kwargs
            )

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    (
                        "Module must have its parameters and buffers on device "
                        "{} but found one of them on device {}."
                    ).format(self.src_device, t.device)
                )

        inputs = [
            batch.to(f"cuda:{self.device_ids[i]}")
            for i, batch in enumerate(batch_list)
        ]
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class ParallelCollater:
    def __init__(self, num_gpus: int, otf_graph: bool = False) -> None:
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph

    def __call__(self, data_list: List[BaseData]) -> List[BaseData]:
        if self.num_gpus in [0, 1]:  # adds cpu-only case
            batch = data_list_collater(data_list, otf_graph=self.otf_graph)
            return [batch]

        else:
            num_devices = min(self.num_gpus, len(data_list))

            count = torch.tensor([data.num_nodes for data in data_list])
            cumsum = count.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = (
                num_devices * cumsum.to(torch.float) / cumsum[-1].item()
            )
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                data_list_collater(data_list[split[i] : split[i + 1]])
                for i in range(len(split) - 1)
            ]


@numba.njit
def balanced_partition(sizes: npt.NDArray[np.int_], num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap: List[Tuple[List[int], List[int]]] = []
    for idx in sort_idx[:num_parts]:
        heap.append((sizes[idx], [idx]))
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + sizes[idx]
        new_idx = smallest_part[1] + [idx]
        heapq.heappush(heap, (new_size, new_idx))
    idx_balanced = [part[1] for part in heap]
    return idx_balanced


@runtime_checkable
class _HasMetadata(Protocol):
    @property
    def metadata_path(self) -> Path:
        ...


class BalancedBatchSampler(Sampler):
    def _load_dataset(self, dataset, mode: Literal["atoms", "neighbors"]):
        errors: List[str] = []
        if not isinstance(dataset, _HasMetadata):
            errors.append(
                f"Dataset {dataset} does not have a metadata_path attribute."
            )
            return None, errors
        if not dataset.metadata_path.exists():
            errors.append(
                f"Metadata file {dataset.metadata_path} does not exist."
            )
            return None, errors

        key = {"atoms": "natoms", "neighbors": "neighbors"}[mode]
        sizes = np.load(dataset.metadata_path)[key]

        return sizes, errors

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: torch.device,
        mode: Union[str, bool] = "atoms",
        shuffle: bool = True,
        drop_last: bool = False,
        force_balancing: bool = False,
        throw_on_error: bool = False,
    ) -> None:
        if mode is True:
            mode = "atoms"

        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ("atoms", "neighbors"):
                raise ValueError(
                    f"Invalid mode {mode}. Must be one of 'atoms', 'neighbors', or a boolean."
                )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.device = device
        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.single_sampler = DistributedSampler(
            self.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_sampler = BatchSampler(
            self.single_sampler,
            batch_size,
            drop_last=drop_last,
        )

        self.sizes = None
        self.balance_batches = False

        if self.num_replicas <= 1:
            logging.info(
                "Batch balancing is disabled for single GPU training."
            )
            return

        if self.mode is False:
            logging.info(
                "Batch balancing is disabled because `optim.load_balancing` is `False`"
            )
            return

        self.sizes, errors = self._load_dataset(dataset, self.mode)
        if self.sizes is None:
            self.balance_batches = force_balancing
            if force_balancing:
                errors.append(
                    "BalancedBatchSampler has to load the data to  determine batch sizes, which incurs significant overhead! "
                    "You can disable balancing by setting `optim.load_balancing` to `False`."
                )
            else:
                errors.append(
                    "Batches will not be balanced, which can incur significant overhead!"
                )
        else:
            self.balance_batches = True

        if errors:
            msg = "BalancedBatchSampler: " + " ".join(errors)
            if throw_on_error:
                raise RuntimeError(msg)
            else:
                logging.warning(msg)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        self.single_sampler.set_epoch(epoch)

    def __iter__(self):
        if not self.balance_batches:
            yield from self.batch_sampler
            return

        for batch_idx in self.batch_sampler:
            if self.sizes is None:
                # Unfortunately, we need to load the data to know the image sizes
                data_list = [self.dataset[idx] for idx in batch_idx]

                if self.mode == "atoms":
                    sizes = [data.num_nodes for data in data_list]
                elif self.mode == "neighbors":
                    sizes = [data.edge_index.shape[1] for data in data_list]
                else:
                    raise NotImplementedError(
                        f"Unknown load balancing mode: {self.mode}"
                    )
            else:
                sizes = [self.sizes[idx] for idx in batch_idx]

            idx_sizes = torch.stack(
                [torch.tensor(batch_idx), torch.tensor(sizes)]
            )
            idx_sizes_all = distutils.all_gather(idx_sizes, device=self.device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            if gp_utils.initialized():
                idx_sizes_all = torch.unique(input=idx_sizes_all, dim=1)
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = balanced_partition(
                sizes_all.numpy(), num_parts=self.num_replicas
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.rank]]
