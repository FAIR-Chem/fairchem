import copy
from collections import abc
from functools import cache
from logging import getLogger
from typing import Any, Callable, cast

import numpy as np
import wrapt
from torch.utils.data import Dataset
from typing_extensions import TypeVar, override

log = getLogger(__name__)

TDataset = TypeVar("TDataset", bound=Dataset, infer_variance=True)


def dataset_transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    copy_data: bool = False,
) -> TDataset:
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

    return cast(TDataset, _TransformedDataset(dataset))


def expand_dataset(dataset: Dataset, n: int) -> Dataset:
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
            log.debug(
                f"Expanded the atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    dataset = cast(Dataset, _ExpandedDataset(dataset))
    log.info(
        f"Expanded dataset {dataset.__class__.__name__} from {og_size} to {n}."
    )
    return dataset
