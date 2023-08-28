from abc import ABC
from logging import getLogger
from typing import TYPE_CHECKING, Any, Mapping, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from typing_extensions import Self, dataclass_transform, override

log = getLogger(__name__)


_ModelBase = BaseModel
if TYPE_CHECKING:
    _ModelBase = ABC


@dataclass_transform(kw_only_default=True)
class TypedConfig(_ModelBase):
    model_config = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
    )

    def __post_init__(self):
        # Override this method to perform any post-initialization
        # actions on the model.
        pass

    if not TYPE_CHECKING:

        def model_post_init(self, __context: Any):
            self.__post_init__()

    @classmethod
    @property
    def _as_pydantic_model_cls(cls):
        return cast(BaseModel, cls)

    @property
    def _as_pydantic_model(self):
        return cast(BaseModel, self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        return cast(Self, cls._as_pydantic_model_cls.model_validate(d))

    def to_dict(self) -> dict[str, Any]:
        return self._as_pydantic_model.model_dump()


class Singleton:
    singleton_key = "_singleton_instance"

    @classmethod
    def get(cls) -> Self | None:
        return getattr(cls, cls.singleton_key, None)

    @classmethod
    def set(cls, instance: Self) -> None:
        if cls.get() is not None:
            log.warning(f"{cls.__qualname__} instance is already set")

        setattr(cls, cls.singleton_key, instance)

    @classmethod
    def reset(cls) -> None:
        if cls.get() is not None:
            delattr(cls, cls.singleton_key)

    @classmethod
    def register(cls, instance: Self) -> None:
        cls.set(instance)

    @classmethod
    def instance(cls) -> Self:
        instance = cls.get()
        if instance is None:
            raise RuntimeError(f"{cls.__qualname__} instance is not set")

        return instance

    @override
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)

        cls.reset()


__all__ = [
    "Field",
    "Singleton",
    "TypeAdapter",
    "TypedConfig",
    "ValidationError",
    "field_validator",
]
