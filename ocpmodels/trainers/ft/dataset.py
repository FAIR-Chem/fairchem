import pickle
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, cast

from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

from ocpmodels.common.utils import apply_key_mapping

from ..mt import dataset_transform as DT
from ..mt.config import ModelConfig, SplitDatasetConfig, TransformConfigs
from ..mt.dataset import (
    MTConcatDataset,
    _create_split_dataset,
    _set_sid_transform,
)
from ..mt.normalizer import normalizer_transform
from .config import FTDatasetsConfig

log = getLogger(__name__)


def _apply_ft_transforms(
    dataset: Dataset[Any],
    config: FTDatasetsConfig,
    split_config: SplitDatasetConfig,
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
    if config.normalization:
        dataset = DT.dataset_transform(
            dataset,
            normalizer_transform(config.normalization),
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


def create_ft_datasets(config: FTDatasetsConfig, model: ModelConfig):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    transform_configs = TransformConfigs(mt=None, model=model)

    # Create the train, val, test datasets
    if config.train is not None:
        train_dataset = _create_split_dataset(config.train.config)
        train_dataset = _apply_ft_transforms(
            train_dataset,
            config,
            config.train,
            transform_configs,
            training=True,
        )
    if config.val is not None:
        datasets: list[Dataset[Any]] = []
        for val_config in config.val:
            dataset = _create_split_dataset(val_config.config)
            dataset = _apply_ft_transforms(
                dataset,
                config,
                val_config,
                transform_configs,
                training=False,
            )
            datasets.append(dataset)
        val_dataset = MTConcatDataset(datasets)
    if config.test is not None:
        datasets: list[Dataset[Any]] = []
        for test_config in config.test:
            dataset = _create_split_dataset(test_config.config)
            dataset = _apply_ft_transforms(
                dataset,
                config,
                test_config,
                transform_configs,
                training=False,
            )
            datasets.append(dataset)
        test_dataset = MTConcatDataset(datasets)

    return train_dataset, val_dataset, test_dataset
