from typing import Generic, Iterable, Iterator, Optional, TypeVar, overload

import torch.nn as nn

TModule = TypeVar("TModule", bound=nn.Module)


class TypedModuleList(nn.ModuleList, Generic[TModule]):
    def __init__(self, modules: Optional[Iterable[TModule]] = None) -> None:
        super().__init__(modules)

    @overload
    def __getitem__(self, idx: int) -> TModule:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[TModule]":
        ...

    def __getitem__(self, idx: int | slice) -> TModule | "TypedModuleList[TModule]":
        return super().__getitem__(idx)  # type: ignore

    def __setitem__(self, idx: int, module: TModule) -> None:
        return super().__setitem__(idx, module)

    def __iter__(self) -> Iterator[TModule]:
        return super().__iter__()  # type: ignore

    def __iadd__(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":
        return super().__iadd__(modules)  # type: ignore

    def __add__(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":
        return super().__add__(modules)  # type: ignore

    def insert(self, idx: int, module: TModule) -> None:
        return super().insert(idx, module)  # type: ignore

    def append(self, module: TModule) -> "TypedModuleList[TModule]":
        return super().append(module)  # type: ignore

    def extend(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":
        return super().extend(modules)  # type: ignore
