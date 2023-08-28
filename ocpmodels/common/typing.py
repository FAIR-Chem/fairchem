from typing import Generic, Iterable, Iterator, Optional, Type, overload

import torch.nn as nn
from typing_extensions import TypeVar, override

_T = TypeVar("_T")


def assert_is_instance(obj: object, cls: Type[_T]) -> _T:
    if not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj


def none_throws(x: Optional[_T], msg: Optional[str] = None) -> _T:
    if x is None:
        if msg:
            raise ValueError(msg)
        else:
            raise ValueError("x cannot be None")
    return x


TModule = TypeVar("TModule", infer_variance=True, bound=nn.Module)


class TypedModuleList(nn.ModuleList, Generic[TModule]):
    @override
    def __init__(self, modules: Optional[Iterable[TModule]] = None) -> None:
        super().__init__(modules)

    @overload
    def __getitem__(self, idx: int) -> TModule:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[TModule]":
        ...

    @override
    def __getitem__(
        self, idx: int | slice
    ) -> TModule | "TypedModuleList[TModule]":
        return super().__getitem__(idx)  # type: ignore

    @override
    def __setitem__(self, idx: int, module: TModule) -> None:
        return super().__setitem__(idx, module)

    @override
    def __iter__(self) -> Iterator[TModule]:
        return super().__iter__()  # type: ignore

    @override
    def __iadd__(
        self, modules: Iterable[TModule]
    ) -> "TypedModuleList[TModule]":
        return super().__iadd__(modules)  # type: ignore

    @override
    def __add__(
        self, modules: Iterable[TModule]
    ) -> "TypedModuleList[TModule]":
        return super().__add__(modules)  # type: ignore

    @override
    def insert(self, idx: int, module: TModule) -> None:
        return super().insert(idx, module)  # type: ignore

    @override
    def append(self, module: TModule) -> "TypedModuleList[TModule]":
        return super().append(module)  # type: ignore

    @override
    def extend(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":
        return super().extend(modules)  # type: ignore
