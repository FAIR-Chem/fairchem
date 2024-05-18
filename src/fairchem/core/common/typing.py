from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def assert_is_instance(obj: object, cls: type[_T]) -> _T:
    if obj and not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj


def none_throws(x: _T | None, msg: str | None = None) -> _T:
    if x is None:
        if msg:
            raise ValueError(msg)
        raise ValueError("x cannot be None")
    return x
