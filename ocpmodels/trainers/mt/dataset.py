import pickle
from collections import abc
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable, List, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from typing_extensions import override

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import apply_key_mapping

from . import dataset_transform as DT
from .balanced_batch_sampler import DatasetWithSizes
from .config import (
    DatasetConfig,
    FullyBalancedSamplingConfig,
    ModelConfig,
    MultiTaskConfig,
    OneHotTargetsConfig,
    SamplingConfig,
    SplitDatasetConfig,
    TaskConfig,
    TaskDatasetConfig,
    TemperatureSamplingConfig,
    TransformConfigs,
)
from .normalizer import normalizer_transform

log = getLogger(__name__)


def _set_sid_transform(dataset: DT.TDataset) -> DT.TDataset:
    def transform_idx(idx: int, data):
        if getattr(data, "sid", None) is None:
            data.sid = torch.tensor([idx], dtype=torch.long)
        return data

    dataset = DT.dataset_idx_transform(dataset, transform_idx)
    return dataset


class MTConcatDataset(ConcatDataset):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        # For the indices, find the dataset that they belong to
        dataset_idx = np.searchsorted(
            self.cumulative_sizes, indices, side="right"
        )
        dataset_idx = np.clip(dataset_idx, 0, len(self.cumulative_sizes) - 1)

        # Get the indices for each dataset
        indices_array = np.array(indices)
        dataset_indices = [
            indices_array[np.where(dataset_idx == idx)[0]]
            for idx in range(len(self.datasets))
        ]

        # Get the sizes for each dataset
        sizes_list: list[np.ndarray] = []
        for idx, dataset in enumerate(self.datasets):
            if not isinstance(dataset, DatasetWithSizes):
                raise TypeError(f"{dataset=} is not a DatasetWithSizes.")

            curr_dataset_indices = dataset_indices[idx]
            if len(curr_dataset_indices) > 0:
                sizes_list.append(dataset.data_sizes(curr_dataset_indices))

        if not sizes_list:
            raise ValueError(f"No sizes found for {indices=}")

        # Concatenate the sizes
        sizes = np.concatenate(sizes_list)
        assert len(sizes) == len(
            indices
        ), f"{sizes=} does not match {indices=}"
        return sizes

    @override
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        for dataset in datasets:
            if not isinstance(dataset, abc.Sized):
                raise TypeError(f"{dataset=} is not a Sized.")

        super().__init__(datasets)


def _create_split_dataset(config: dict[str, Any]) -> Dataset:
    # Create the dataset
    dataset_cls = registry.get_dataset_class(config["format"])
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls=} is not a Dataset"
    dataset = cast(Any, dataset_cls)(config)
    dataset = cast(Dataset, dataset)

    return dataset


def _mt_taskify_transform(
    dataset: Dataset[Any],
    task_config: TaskConfig,
    total_num_tasks: int,
    one_hot_targets: OneHotTargetsConfig,
):
    def _transform(data: Data):
        # Wrap the dataset with task_idx transform
        nonlocal task_config, total_num_tasks, one_hot_targets

        if not isinstance(data, Data):
            raise TypeError(f"{data=} is not a torch_geometric.data.Data")

        # Set the task_idx
        data.task_idx = torch.tensor(task_config.idx, dtype=torch.long)
        onehot: torch.Tensor = F.one_hot(
            data.task_idx, num_classes=total_num_tasks
        ).bool()  # (t,)
        # Set task boolean mask
        data.task_mask = rearrange(onehot, "t -> 1 t")

        return data

    dataset = DT.dataset_transform(dataset, _transform)
    return dataset


def _apply_transforms(
    dataset: Dataset[Any],
    config: TaskDatasetConfig,
    split_config: SplitDatasetConfig,
    task_config: TaskConfig,
    total_num_tasks: int,
    one_hot_targets: OneHotTargetsConfig,
    transform_configs: TransformConfigs,
    training: bool,
):
    # Key mapping transform
    if config.key_mapping:
        dataset = DT.dataset_transform(
            dataset,
            partial(apply_key_mapping, key_mapping=config.key_mapping),
        )

    # Set sid if not present
    dataset = _set_sid_transform(dataset)

    # Referencing transform
    if config.referencing:
        if isinstance(config.referencing, Path):
            with config.referencing.open("rb") as f:
                referencing = pickle.load(f)

            assert isinstance(
                referencing, dict
            ), f"Referencing must be a dict, got {type(referencing)}"
        else:
            referencing = config.referencing
        dataset = DT.referencing_transform(dataset, cast(Any, referencing))

    # Normalization transform
    if task_config.normalization:
        dataset = DT.dataset_transform(
            dataset,
            normalizer_transform(task_config.normalization),
        )

    # Taskify/onehot transform
    dataset = _mt_taskify_transform(
        dataset,
        task_config,
        total_num_tasks,
        one_hot_targets,
    )

    # first_n/sample_n transform
    if (first_n := split_config.first_n) is not None:
        dataset = DT.first_n_transform(dataset, first_n.n)
    if (sample_n := split_config.sample_n) is not None:
        dataset = DT.sample_n_transform(dataset, sample_n.n, sample_n.seed)

    # Additonal transforms from the config
    if config.mt_transform_fn is not None:
        dataset = DT.dataset_transform(
            dataset,
            partial(
                config.mt_transform_fn,
                config=transform_configs,
                training=training,
            ),
        )

    return dataset


def _create_task_datasets(
    config: TaskDatasetConfig,
    task_config: TaskConfig,
    total_num_tasks: int,
    one_hot_targets: OneHotTargetsConfig,
    transform_configs: TransformConfigs,
):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Create the train, val, test datasets
    if config.train is not None:
        train_dataset = _create_split_dataset(config.train.config)
        train_dataset = _apply_transforms(
            train_dataset,
            config,
            config.train,
            task_config,
            total_num_tasks,
            one_hot_targets,
            transform_configs=transform_configs,
            training=True,
        )
    if config.val is not None:
        datasets: list[Dataset[Any]] = []
        for val_config in config.val:
            dataset = _create_split_dataset(val_config.config)
            dataset = _apply_transforms(
                dataset,
                config,
                val_config,
                task_config,
                total_num_tasks,
                one_hot_targets,
                transform_configs=transform_configs,
                training=False,
            )
            datasets.append(dataset)
        val_dataset = MTConcatDataset(datasets)
    if config.test is not None:
        datasets: list[Dataset[Any]] = []
        for test_config in config.test:
            dataset = _create_split_dataset(test_config.config)
            dataset = _apply_transforms(
                dataset,
                config,
                test_config,
                task_config,
                total_num_tasks,
                one_hot_targets,
                transform_configs=transform_configs,
                training=False,
            )
            datasets.append(dataset)
        test_dataset = MTConcatDataset(datasets)

    return train_dataset, val_dataset, test_dataset


def _merged_dataset(dataset_sizes_list: List[int], ratios_list: List[float]):
    dataset_sizes = np.array(dataset_sizes_list)
    ratios = np.array(ratios_list)

    # Calculate the target size of the final dataset
    target_size = sum(dataset_sizes) / sum(ratios)

    # Calculate the minimum expansion factor for each dataset
    expansion_factors = target_size * ratios / dataset_sizes

    # Make sure that the expansion factors are all at least 1.0
    expansion_factors = expansion_factors / np.min(expansion_factors)

    # Calculate the number of samples to take from each dataset
    samples_per_dataset = np.ceil(
        dataset_sizes * (expansion_factors / np.min(expansion_factors))
    ).astype(int)

    samples_per_dataset = cast(List[int], samples_per_dataset.tolist())
    return samples_per_dataset


def _combine_datasets(sampling: SamplingConfig, datasets: List[Dataset]):
    # Make sure all datasets have sizes
    dataset_sizes: List[int] = []
    for dataset in datasets:
        if not isinstance(dataset, abc.Sized):
            raise TypeError(f"{dataset=} is not a Sized")
        dataset_sizes.append(len(dataset))

    if isinstance(sampling, FullyBalancedSamplingConfig):
        ratios = [1.0] * len(dataset_sizes)
    elif isinstance(sampling, TemperatureSamplingConfig):
        total_size = sum(dataset_sizes)
        ratios = [
            (size / total_size) ** (1.0 / sampling.temperature)
            for size in dataset_sizes
        ]
    else:
        raise NotImplementedError(f"{sampling=} not implemented.")

    # Normalize the ratios
    ratios = [r / sum(ratios) for r in ratios]
    log.info(f"Using {ratios=} for {sampling=}.")

    # Calculate the expanded dataset sizes
    expanded_dataset_sizes = _merged_dataset(dataset_sizes, ratios)

    # Expand the datasets
    expanded_datasets = [
        DT.expand_dataset(d, n)
        for d, n in zip(datasets, expanded_dataset_sizes)
    ]

    # Combine the datasets
    combined_dataset = MTConcatDataset(expanded_datasets)
    log.info(
        f"Combined {len(expanded_datasets)} datasets into {len(combined_dataset)}."
    )
    return combined_dataset, expanded_dataset_sizes


def create_datasets(
    config: DatasetConfig,
    multi_task: MultiTaskConfig,
    model: ModelConfig,
):
    total_num_tasks = len(config.datasets)
    assert total_num_tasks > 0, "No tasks found in the config."

    transform_configs = TransformConfigs(mt=multi_task, model=model)

    # Create all the datasets
    train_datasets: List[Dataset] = []
    val_datasets: List[Dataset] = []
    test_datasets: List[Dataset] = []
    for task_idx, task_dataset_config in enumerate(config.datasets):
        task_config = multi_task.task_by_idx(task_idx)

        train_dataset, val_dataset, test_dataset = _create_task_datasets(
            task_dataset_config,
            task_config,
            total_num_tasks,
            config.one_hot_targets,
            transform_configs=transform_configs,
        )

        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)
        if test_dataset is not None:
            test_datasets.append(test_dataset)

    # Combine the datasets
    # For train, we need to adhere to the sampling strategy
    train_dataset = None
    train_dataset_sizes = None
    if train_datasets:
        train_dataset, train_dataset_sizes = _combine_datasets(
            config.sampling, train_datasets
        )

    # For val and test, we just concatenate them
    val_dataset = MTConcatDataset(val_datasets) if val_datasets else None
    if val_dataset:
        log.info(f"Created {len(val_dataset)} val samples.")
    test_dataset = MTConcatDataset(test_datasets) if test_datasets else None
    if test_dataset:
        log.info(f"Created {len(test_dataset)} test samples.")

    return (train_dataset, val_dataset, test_dataset), train_dataset_sizes
