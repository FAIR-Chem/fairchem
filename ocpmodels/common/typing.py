from typing import Optional, Type, TypeVar

_T = TypeVar("_T")


def assert_is_instance(obj: object, cls: Type[_T]) -> _T:
    if obj and not isinstance(obj, cls):
        raise TypeError(f"obj is not an instance of cls: obj={obj}, cls={cls}")
    return obj


def none_throws(x: Optional[_T], msg: Optional[str] = None) -> _T:
    if x is None:
        if msg:
            raise ValueError(msg)
        else:
            raise ValueError("x cannot be None")
    return x
