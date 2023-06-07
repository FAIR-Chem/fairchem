from collections import abc
from logging import getLogger
from typing import Any, Callable, Iterable, Sequence, Sized, cast

import wrapt
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar, deprecated, override

log = getLogger(__name__)

TDataset = TypeVar("TDataset", bound=Dataset, infer_variance=True)


class DatasetModuleMixin:
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dataset_transforms: list[Callable[[Any], Any]] = []

    def register_data_transform(self, transform: Callable[[Any], Any]):
        self._dataset_transforms.append(transform)

    def unregister_data_transform(self, transform: Callable[[Any], Any]):
        self._dataset_transforms.remove(transform)

    def _ll_data_transform(self):
        def _transform(x: Any):
            nonlocal self

            for transform in self._dataset_transforms:
                x = transform(x)
            return x

        return _transform

    @property
    def __trainer(self):
        if not isinstance(self, (LightningModule, LightningDataModule)):
            return None
        try:
            return self.trainer
        except RuntimeError:
            return None

    @deprecated(
        "self.dataset(...) is deprecated, use self.register_data_transform(...) instead"
    )
    def dataset(
        self,
        dataset: TDataset,
        name: str | None = None,
        *,
        transform: Callable[[Any], Any] | None = None,
        transform_idx: Callable[[Any, Any], Any] | None = None,
        first_n: int | None = None,
    ) -> TDataset:
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be a torch.utils.data.Dataset, got {type(dataset)}"
            )
        if first_n is not None and not isinstance(dataset, abc.Sized):
            raise TypeError(
                f"first_n is only supported for datasets with a __len__ method, got {type(dataset)}"
            )

        dataset = self.log_dataset_info(dataset, name)

        # create a proxy object around dataset and override __getitem__ to apply transform
        class _DatasetProxy(wrapt.ObjectProxy):
            @override
            def __len__(self):
                if first_n is not None:
                    return min(first_n, len(self.__wrapped__))
                return len(self.__wrapped__)

            @override
            def __getitem__(self, index):
                nonlocal first_n, transform, transform_idx

                if first_n is not None:
                    if not isinstance(index, int):
                        raise TypeError(
                            f"first_n is only supported for integer indices, got {type(index)}"
                        )
                    index = index % first_n

                item = self.__wrapped__[index]
                if transform is not None:
                    item = transform(item)
                if transform_idx is not None:
                    item = transform_idx(item, index)
                return item

        proxy = _DatasetProxy(dataset)
        assert isinstance(proxy, Dataset), f"proxy is not a Dataset: {type(proxy)}"
        return cast(TDataset, proxy)

    def dataloader(
        self,
        dataset: Dataset,
        *,
        num_workers: int,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        pin_memory: bool = False,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
    ):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be a torch.utils.data.Dataset, got {type(dataset)}"
            )

        transform = self._ll_data_transform()

        class _DatasetProxy(wrapt.ObjectProxy):
            @override
            def __getitem__(self, index):
                nonlocal transform

                item = self.__wrapped__[index]
                item = transform(item)
                return item

        dataset = cast(Dataset, _DatasetProxy(dataset))

        if sampler or batch_sampler:
            if (
                self.__trainer is not None
                and self.__trainer._accelerator_connector.use_distributed_sampler
            ):
                raise ValueError(
                    "Trainer.use_distributed_sampler must be False to use custom sampler/batch_sampler"
                )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            sampler=sampler,
            batch_sampler=batch_sampler,
        )

        if isinstance(dataset, Sized):
            log.info(
                f"Created dataloader with: {dataloader.batch_size=}, {dataloader.num_workers=}, {len(dataloader)=}, {len(dataset)=}"
            )
        else:
            log.info(
                f"Created dataloader with: {dataloader.batch_size=}, {dataloader.num_workers=}"
            )
        return dataloader

    def distributed_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if self.__trainer is None:
            raise RuntimeError("trainer is None")
        return DistributedSampler(
            dataset,
            num_replicas=self.__trainer.world_size,
            rank=self.__trainer.global_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def log_dataset_info(
        self,
        dataset: TDataset,
        name: str | None = None,
    ) -> TDataset:
        if name is not None:
            dataset_name = f"Dataset {name} ({dataset.__class__.__module__}.{dataset.__class__.__qualname__})"
        else:
            dataset_name = f"Dataset {dataset.__class__.__module__}.{dataset.__class__.__qualname__}"

        if isinstance(dataset, Sized):
            log.info(f"{dataset_name} loaded with {len(dataset)} samples.")
        else:
            log.info(f"{dataset_name} loaded.")

        return dataset
