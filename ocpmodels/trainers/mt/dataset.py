import logging
from collections import abc
from typing import Any, Literal, Union, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data
from typing_extensions import Annotated, override

from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import Field, TypedConfig

from .transform import dataset_transform, expand_dataset


class SplitDatasetConfig(TypedConfig):
    format: str
    src: str

    key_mapping: dict[str, str] = {}
    transforms: list[Any] = []


class TaskDatasetConfig(TypedConfig):
    train: SplitDatasetConfig | None = None
    val: SplitDatasetConfig | None = None
    test: SplitDatasetConfig | None = None

    copy_from_train: bool = True

    @override
    def model_post_init(self, __context: Any):
        super().model_post_init(__context)

        if self.copy_from_train and self.train is not None:
            if self.val is not None:
                if not self.val.key_mapping:
                    self.val.key_mapping = self.train.key_mapping.copy()
                if not self.val.transforms:
                    self.val.transforms = self.train.transforms.copy()
            if self.test is not None:
                if not self.test.key_mapping:
                    self.test.key_mapping = self.train.key_mapping.copy()
                if not self.test.transforms:
                    self.test.transforms = self.train.transforms.copy()


class TemperatureSamplingConfig(TypedConfig):
    type: Literal["temperature"] = "temperature"
    temperature: float


class FullyBalancedSamplingConfig(TypedConfig):
    type: Literal["fully_balanced"] = "fully_balanced"


SamplingConfig = Annotated[
    Union[
        TemperatureSamplingConfig,
        FullyBalancedSamplingConfig,
    ],
    Field(discriminator="type"),
]


class DatasetConfig(TypedConfig):
    datasets: list[TaskDatasetConfig] = []
    sampling: SamplingConfig = TemperatureSamplingConfig(temperature=1.0)


def _create_split_dataset(
    config: SplitDatasetConfig,
    task_idx: int,
    total_num_tasks: int,
) -> Dataset:
    # Create the dataset
    dataset_cls = registry.get_dataset_class(config.format)
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls=} is not a Dataset"
    dataset = cast(Any, dataset_cls)(config.to_dict())
    dataset = cast(Dataset, dataset)

    # Wrap the dataset with task_idx transform
    def _transform(data: Data):
        if not isinstance(data, Data):
            raise TypeError(f"{data=} is not a torch_geometric.data.Data")

        # Set the task_idx
        data.task_idx = torch.tensor(task_idx, dtype=torch.long)
        # Set the task_mask as a one-hot vector
        data.task_mask = torch.zeros(total_num_tasks, dtype=torch.bool)
        data.task_mask[task_idx] = True
        data.task_mask = data.task_mask.unsqueeze(dim=0)
        return data

    dataset = dataset_transform(dataset, _transform)
    return dataset


def _create_task_datasets(
    config: TaskDatasetConfig,
    task_idx: int,
    total_num_tasks: int,
):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Create the train, val, test datasets
    if config.train is not None:
        train_dataset = _create_split_dataset(
            config.train, task_idx, total_num_tasks
        )
    if config.val is not None:
        val_dataset = _create_split_dataset(
            config.val, task_idx, total_num_tasks
        )
    if config.test is not None:
        test_dataset = _create_split_dataset(
            config.test, task_idx, total_num_tasks
        )
    return train_dataset, val_dataset, test_dataset


def _merged_dataset(dataset_sizes_list: list[int], ratios_list: list[float]):
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

    samples_per_dataset = cast(list[int], samples_per_dataset.tolist())
    return samples_per_dataset


def _combine_datasets(sampling: SamplingConfig, datasets: list[Dataset]):
    # Make sure all datasets have sizes
    dataset_sizes: list[int] = []
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
    logging.info(f"Using {ratios=} for {sampling=}.")

    # Calculate the expanded dataset sizes
    expanded_dataset_sizes = _merged_dataset(dataset_sizes, ratios)

    # Expand the datasets
    expanded_datasets = [
        expand_dataset(d, n) for d, n in zip(datasets, expanded_dataset_sizes)
    ]

    # Combine the datasets
    combined_dataset = ConcatDataset(expanded_datasets)
    logging.info(
        f"Combined {len(expanded_datasets)} datasets into {len(combined_dataset)}."
    )
    return combined_dataset


def create_datasets(config: DatasetConfig):
    total_num_tasks = len(config.datasets)
    assert total_num_tasks > 0, "No tasks found in the config."

    # Create all the datasets
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    for task_idx, task_dataset_config in enumerate(config.datasets):
        train_dataset, val_dataset, test_dataset = _create_task_datasets(
            task_dataset_config, task_idx, total_num_tasks
        )

        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)
        if test_dataset is not None:
            test_datasets.append(test_dataset)

    # Combine the datasets
    # For train, we need to adhere to the sampling strategy
    train_dataset = (
        _combine_datasets(config.sampling, train_datasets)
        if train_datasets
        else None
    )

    # For val and test, we just concatenate them
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    return train_dataset, val_dataset, test_dataset
