import inspect
import json
from abc import ABC
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic.fields import FieldInfo
from typing_extensions import Self, dataclass_transform, override

from ._docs_extraction import extract_docstrings_from_cls

log = getLogger(__name__)


_ModelBase = BaseModel
if TYPE_CHECKING:
    _ModelBase = ABC


def _update_fields_from_docstrings(
    fields: dict[str, FieldInfo],
    fields_docs: dict[str, str],
) -> None:
    for ann_name, field_info in fields.items():
        if field_info.description is None and ann_name in fields_docs:
            field_info.description = fields_docs[ann_name]


@dataclass_transform(kw_only_default=True)
class TypedConfig(_ModelBase):
    @override
    @classmethod
    def __init_subclass__(
        cls,
        use_attributes_docstring: bool = True,
        write_schema_to_file: bool = False,
    ):
        super().__init_subclass__()

    @classmethod
    def __update_description(cls):
        description_parts: list[str] = []

        # Add the class docstring if it exists.
        if cls.__doc__:
            description_parts.append(cls.__doc__.strip())

        cls._as_pydantic_model_cls.model_config["json_schema_extra"] = {
            "description": "\n".join(description_parts)
        }

    @classmethod
    def __pydantic_init_subclass__(
        cls,
        use_attributes_docstring: bool = True,
        write_schema_to_file: bool = False,
    ):
        super().__pydantic_init_subclass__()  # type: ignore

        # Set the description of the model to the class docstring + some additional info.
        cls.__update_description()

        # Update the fields descriptions from the docstrings.
        if use_attributes_docstring:
            fields_docs = getattr(cls, "fields_docs", None)
            if fields_docs is None:
                fields_docs = {}
            fields_docs = {**fields_docs, **extract_docstrings_from_cls(cls)}
            setattr(cls, "fields_docs", fields_docs)

            _update_fields_from_docstrings(
                cls._as_pydantic_model_cls.model_fields, fields_docs
            )

        # If requested, write the schema to a file.
        if write_schema_to_file:
            cls_file_path = inspect.getfile(cls)
            if cls_file_path:
                # Save the schema to a file with the same name as the class.
                dest = Path(cls_file_path).with_suffix(
                    f".{cls.__name__}.schema.json"
                )
                cls.dump_schema(dest)

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

    @classmethod
    def dump_schema(cls, dest: Path):
        _ = cls._as_pydantic_model_cls.model_rebuild(force=True)
        json_schema = cls._as_pydantic_model_cls.model_json_schema()
        _ = dest.write_text(json.dumps(json_schema, indent=4))

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
