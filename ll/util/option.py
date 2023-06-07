from typing import Callable

from typing_extensions import TypeVar

T = TypeVar("T", infer_variance=True)


def default_error_fn():
    return ValueError("Value must be defined")


def ensure_defined(
    value: T | None,
    *,
    error_fn: Callable[[], Exception] = default_error_fn,
) -> T:
    if value is None:
        raise error_fn()
    return value
