import copy
import logging
from collections import abc
from functools import cache
from typing import Any, Callable, Literal, Union, cast

import numpy as np
import pydantic
import torch
import wrapt
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data
from typing_extensions import Annotated, override

from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry

from .ocp_trainer import OCPTrainer


class IdentityTransformConfig(pydantic.BaseModel):
    type: Literal["identity"] = "identity"


class NormalizerTransformConfig(pydantic.BaseModel):
    type: Literal["normalizer"] = "normalizer"


TransformConfig = Annotated[
    Union[IdentityTransformConfig, NormalizerTransformConfig],
    pydantic.Field(discriminator="type"),
]


class SplitDatasetConfig(pydantic.BaseModel):
    format: str = "lmdb"
    src: str = ""

    key_mapping: dict[str, str] = {}
    transforms: list[TransformConfig] = []


class TaskDatasetConfig(pydantic.BaseModel):
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


class TemperatureSamplingConfig(pydantic.BaseModel):
    type: Literal["temperature"] = "temperature"
    temperature: float = 1.0


class FullyBalancedSamplingConfig(pydantic.BaseModel):
    type: Literal["fully_balanced"] = "fully_balanced"


SamplingConfig = Annotated[
    Union[
        TemperatureSamplingConfig,
        FullyBalancedSamplingConfig,
    ],
    pydantic.Field(discriminator="type"),
]


class DatasetConfig(pydantic.BaseModel):
    datasets: list[TaskDatasetConfig] = []
    sampling: SamplingConfig = TemperatureSamplingConfig()


def _dataset_transform(
    dataset: Dataset,
    transform: Callable[[Any], Any],
    copy_data: bool = False,
) -> Dataset:
    class _TransformedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx):
            nonlocal copy_data, transform

            assert transform is not None, "Transform must be defined."
            data = self.__wrapped__.__getitem__(idx)
            if copy_data:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(Dataset, _TransformedDataset(dataset))


def _expand_dataset(dataset: Dataset, n: int) -> Dataset:
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"expand_dataset ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    og_size = len(dataset)
    if og_size > n:
        raise ValueError(
            f"expand_dataset ({n}) must be greater than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__}"
        )

    class _ExpandedDataset(wrapt.ObjectProxy):
        @override
        def __len__(self):
            nonlocal n
            return n

        @override
        def __getitem__(self, index: int):
            nonlocal n, og_size
            if index < 0 or index >= n:
                raise IndexError(
                    f"Index {index} is out of bounds for dataset of size {n}."
                )
            return self.__wrapped__.__getitem__(index % og_size)

        @cache
        def _atoms_metadata_cached(self):
            """
            We want to retrieve the atoms metadata for the expanded dataset.
            This includes repeating the atoms metadata for the elemens that are repeated.
            """

            # the out metadata shape should be (n,)
            nonlocal n, og_size

            metadata = self.__wrapped__.atoms_metadata
            metadata = np.resize(metadata, (n,))
            logging.debug(
                f"Expanded the atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    dataset = cast(Dataset, _ExpandedDataset(dataset))
    logging.info(
        f"Expanded dataset {dataset.__class__.__name__} from {og_size} to {n}."
    )
    return dataset


def _create_split_dataset(
    config: SplitDatasetConfig,
    task_idx: int,
    total_num_tasks: int,
) -> Dataset:
    # Create the dataset
    dataset_cls = registry.get_dataset_class(config.format)
    assert issubclass(dataset_cls, Dataset), f"{dataset_cls=} is not a Dataset"
    dataset = cast(Any, dataset_cls)(dict(config))
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

    dataset = _dataset_transform(dataset, _transform)
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
        _expand_dataset(d, n) for d, n in zip(datasets, expanded_dataset_sizes)
    ]

    # Combine the datasets
    combined_dataset = ConcatDataset(expanded_datasets)
    logging.info(
        f"Combined {len(expanded_datasets)} datasets into {len(combined_dataset)}."
    )
    return combined_dataset


def _create_datasets(config: DatasetConfig):
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


@registry.register_trainer("mt_trainer")
class MTTrainer(OCPTrainer):
    @override
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="ocp",
    ):
        super().__init__(
            task,
            model,
            outputs,
            pydantic.TypeAdapter(DatasetConfig).validate_python(
                dataset
            ),  # HACK: wrap it in a class so it doesn't get registered as a dict
            optimizer,
            loss_fns,
            eval_metrics,
            identifier,
            timestamp_id,
            run_dir,
            is_debug,
            print_every,
            seed,
            logger,
            local_rank,
            amp,
            cpu,
            slurm,
            noddp,
            name,
        )

    @override
    def load_datasets(self) -> None:
        logging.info(
            f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
        )
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )

        batch_size = self.config["optim"]["batch_size"]
        assert isinstance(batch_size, int), f"{batch_size=} is not an integer"
        eval_batch_size = self.config["optim"].get(
            "eval_batch_size", batch_size
        )
        assert isinstance(
            eval_batch_size, int
        ), f"{eval_batch_size=} is not an integer"

        dataset_config = self.config["dataset"]
        assert isinstance(
            dataset_config, DatasetConfig
        ), f"{dataset_config=} is not a DatasetConfig"
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = _create_datasets(dataset_config)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.train_dataset is not None:
            self.train_sampler = self.get_sampler(
                self.train_dataset, batch_size, shuffle=True
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset, self.train_sampler
            )

        if self.val_dataset is not None:
            self.val_sampler = self.get_sampler(
                self.val_dataset, eval_batch_size, shuffle=False
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset, self.val_sampler
            )

        if self.test_dataset is not None:
            self.test_sampler = self.get_sampler(
                self.test_dataset, eval_batch_size, shuffle=False
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset, self.test_sampler
            )
        # load relaxation dataset
        if "relax_dataset" in self.config["task"]:
            raise NotImplementedError(
                "Relaxation dataset not implemented for MT."
            )
