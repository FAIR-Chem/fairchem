"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any, Literal

import numba
import numpy as np
import torch
import torch.distributed
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
from typing_extensions import deprecated, override

from fairchem.core.common import distutils, gp_utils
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.base_dataset import (
    UnsupportedDatasetError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch_geometric.data import Batch, Data


@deprecated(
    "OCPColatter is deprecated. Please use data_list_collater optionally with functools.partial to set defaults"
)
class OCPCollater:
    def __init__(self, otf_graph: bool = False) -> None:
        self.otf_graph = otf_graph

    def __call__(self, data_list: list[Data]) -> Batch:
        return data_list_collater(data_list, otf_graph=self.otf_graph)


@numba.njit
def _balanced_partition(sizes: NDArray[np.int_], num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap = [(sizes[idx], [idx]) for idx in sort_idx[:num_parts]]
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + sizes[idx]
        new_idx = smallest_part[1] + [
            idx
        ]  # TODO should this be append to save time/space
        heapq.heappush(heap, (new_size, new_idx))
    return [part[1] for part in heap]


class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset=dataset, **kwargs)

        self.start_iter = 0
        self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size not set for the sampler"
        logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # TODO: For very large datasets, even virtual datasets this might slow down
        # or not work correctly. The issue is that we enumerate the full list of all
        # samples in a single epoch, and manipulate this list directly. A better way
        # of doing this would be to keep this sequence strictly as an iterator
        # that stores the current state (instead of the full sequence)
        distributed_sampler_sequence = super().__iter__()
        if self.start_iter > 0:
            for i, _ in enumerate(distributed_sampler_sequence):
                if i == self.start_iter * self.batch_size - 1:
                    break
        return distributed_sampler_sequence

    def set_epoch_and_start_iteration(self, epoch, start_iter):
        self.set_epoch(epoch)
        self.start_iter = start_iter


def _ensure_supported(dataset: Any):
    if not isinstance(dataset, Dataset):
        raise UnsupportedDatasetError("BalancedBatchSampler requires a dataset.")

    if not dataset.metadata_hasattr("natoms"):
        raise UnsupportedDatasetError(
            "BalancedBatchSampler requires a dataset that has a metadata attributed with number of atoms."
        )

    logging.debug(f"BalancedBatchSampler: Resolved dataset to {type(dataset)}")
    return dataset


class BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: (
            torch.device | None
        ) = None,  # deprecated, unused variable for backwards compat
        seed: int,
        mode: bool | Literal["atoms"] = "atoms",
        shuffle: bool = True,
        on_error: Literal["warn_and_balance", "warn_and_no_balance", "raise"] = "raise",
        drop_last: bool = False,
    ):
        """
        Initializes a BalancedBatchSampler object.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of each batch.
            num_replicas (int): The number of processes participating in distributed training.
            rank (int): The rank of the current process in distributed training.
            device (torch.device): The device to use for the batches.
            mode (str or bool, optional): The mode to use for balancing the batches. Defaults to "atoms".
            shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.
            on_error (Literal["warn_and_balance", "warn_and_no_balance", "raise"], optional): The action to take when an error occurs (i.e., when we have an invalid dataset). Defaults to "raise".
                - "warn_and_balance": Raise a warning and balance the batch by manually loading the data samples and counting the number of nodes (this is slow).
                - "warn_and_no_balance": Raise a warning and do not do any balancing.
                - "raise": Raise an error.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        """
        self.disabled = False
        self.on_error = on_error

        if mode is False:
            logging.warning(f"Disabled BalancedBatchSampler because {mode=}.")
            self.disabled = True
        elif mode.lower() != "atoms":
            raise ValueError(
                f"Only mode='atoms' or mode=True is supported, got {mode=}."
            )
        elif num_replicas == 1:
            logging.warning(f"Disabled BalancedBatchSampler because {num_replicas=}.")
            self.disabled = True

        try:
            dataset = _ensure_supported(dataset)
        except UnsupportedDatasetError as error:
            if self.on_error == "raise":
                raise error
            if self.on_error == "warn_and_balance":
                logging.warning(
                    f"Failed to get data sizes from metadata, loading data to get sizes (THIS IS SLOW). {error}"
                )
            elif self.on_error == "warn_and_no_balance":
                logging.warning(
                    f"Failed to get data sizes, falling back to uniform partitioning. {error}"
                )
            else:
                raise ValueError(f"Unknown on_error={self.on_error}") from error

        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            batch_size=batch_size,
            seed=seed,
        )

        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)
        self.device = (
            device if device is not None else distutils.get_device_for_local_rank()
        )

        logging.info(
            f"Created BalancedBatchSampler with {sampler=}, {batch_size=}, {drop_last=}"
        )

    def _get_natoms(self, batch_idx: list[int]):
        if self.sampler.dataset.metadata_hasattr("natoms"):
            return np.array(
                self.sampler.dataset.get_metadata("natoms", batch_idx)
            ).reshape(-1)
        if self.on_error == "warn_and_balance":
            return np.array([self.sampler.dataset[idx].num_nodes for idx in batch_idx])
        return None

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int) -> None:
        if not isinstance(self.sampler, StatefulDistributedSampler):
            if start_iteration != 0:
                raise NotImplementedError(
                    f"{type(self.single_sampler)} does not support resuming from a nonzero step."
                )
            self.sampler.set_epoch(epoch)
        else:
            self.sampler.set_epoch_and_start_iteration(epoch, start_iteration)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    @staticmethod
    def _dist_enabled():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @override
    def __iter__(self):
        if self.disabled or not self._dist_enabled():
            yield from super().__iter__()
            return

        for batch_idx in super().__iter__():
            sizes = self._get_natoms(batch_idx)
            if sizes is None:  # on_error == "warn_and_no_balance" is set
                yield batch_idx
                continue

            idx_sizes = torch.stack(
                [
                    torch.tensor(batch_idx, device=self.device),
                    torch.tensor(sizes, device=self.device),
                ]
            )
            idx_sizes_all = distutils.all_gather(idx_sizes, device=self.device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            if gp_utils.initialized():
                idx_sizes_all = torch.unique(input=idx_sizes_all, dim=1)
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = _balanced_partition(
                sizes_all.numpy(),
                num_parts=self.sampler.num_replicas,
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.sampler.rank]]
