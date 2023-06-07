from typing import TYPE_CHECKING, Any, Callable, Concatenate, Type, cast

from typing_extensions import ParamSpec, TypeVar

P = ParamSpec("P")
TSelf = TypeVar("TSelf", infer_variance=True)
TParam = TypeVar("TParam", infer_variance=True)
TReturn = TypeVar("TReturn", infer_variance=True)


def copy_args(
    kwargs_call: Callable[P, Any],
    *,
    return_type: Type[TReturn],
) -> Callable[[Callable[..., TReturn]], Callable[P, TReturn]]:
    """
    Copies the type annotations from one function to another.
    """

    def return_func(func: Callable[..., TReturn]):
        return cast(Callable[P, TReturn], func)

    return return_func


def copy_method_with_param(
    kwargs_call: Callable[Concatenate[TSelf, P], Any],
    *,
    param_type: Type[TParam],
    return_type: Type[TReturn],
) -> Callable[
    [Callable[..., TReturn]], Callable[Concatenate[TSelf, TParam, P], TReturn]
]:
    """
    Copies the type annotations from one method to another,
    but adds a new parameter to the beginning.
    """

    def return_func(func: Callable[..., TReturn]):
        return cast(Callable[Concatenate[TSelf, TParam, P], TReturn], func)

    return return_func


TBase = TypeVar("TBase")


def mixin_base_type(base_class: Type[TBase]) -> Type[TBase]:
    """
    Useful function to make mixins with baseclass typehint

    ```
    class ReadonlyMixin(mixin_base_type(BaseAdmin))):
        ...
    ```
    """
    if TYPE_CHECKING:
        return base_class
    return object
