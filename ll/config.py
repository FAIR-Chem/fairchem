import copy
from abc import ABCMeta
from dataclasses import asdict, dataclass, fields
from functools import wraps
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping

import simplejson
import yaml
from pydantic import Field, parse_obj_as, validator
from pydantic.dataclasses import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.json import pydantic_encoder
from typing_extensions import TypeVar, dataclass_transform, override

from .util.config_dataclasses import monkey_patch_dataclasses

log = getLogger(__name__)

monkey_patch_dataclasses()

# re-export UNDEFINED with the ANY type so that we can use it
# for fields that don't have a default value
UNDEFINED: Any = object()

TCommonValue = TypeVar("TCommonValue", infer_variance=True)


def _copy_dataclass_fields(object: Any):
    for field in fields(object):
        # get the field value
        if (value := getattr(object, field.name, None)) is None or value is UNDEFINED:
            continue

        # set the value to a deep copy of the current value
        setattr(object, field.name, copy.deepcopy(value))


@dataclass_transform(eq_default=True, kw_only_default=True)
class TypedConfigMeta(ABCMeta):
    def __new__(cls, *args, **kwargs):
        # before everything, we check if omegaconf is installed
        # if so, we throw an error because Pytorch Lightning's
        # omegaconf integration is broken
        try:
            import omegaconf  # type: ignore
        except ImportError:
            pass
        else:
            raise ImportError(
                "omegaconf is installed, which is incompatible with TypedConfig. "
                "Please uninstall omegaconf."
            )

        config_cls: Any = super().__new__(cls, *args, **kwargs)

        # create the dataclass
        dataclass_defaults = getattr(config_cls, "__dataclass_transform__", {})
        dataclass_kwargs = dict(
            eq=dataclass_defaults.get("eq_default", True),
            kw_only=dataclass_defaults.get("kw_only_default", True),
            order=dataclass_defaults.get("order_default", False),
            frozen=dataclass_defaults.get("frozen_default", False),
        )
        config_cls = dataclass(**dataclass_kwargs)(config_cls)

        # update the init/post_init method to copy over the fields
        # this allows us to skip having to use the
        # field(default_factory=...) pattern
        if (orig_post_init := getattr(config_cls, "__post_init__", None)) is not None:

            @wraps(orig_post_init)
            def post_init(self, *args, **kwargs):
                # copy the fields first
                _copy_dataclass_fields(self)

                orig_post_init(self, *args, **kwargs)

            config_cls.__post_init__ = post_init
        else:
            # use init method instead
            orig_init = config_cls.__init__

            @wraps(orig_init)
            def init(self, *args, **kwargs):
                # call the original init method first
                orig_init(self, *args, **kwargs)
                _copy_dataclass_fields(self)

            config_cls.__init__ = init

        # Setup pydantic dataclass.
        # We use a pydantic wrapped dataclass instead of a pydantic BaseModel
        # because we want to delay the validation. This is only supported by
        # pydantic wrapped native dataclasses.
        config_cls.__pydantic_wrapped_dataclass__ = pydantic_dataclass(
            config_cls,
            **dataclass_kwargs,
            config=ConfigDict(
                validate_assignment=True,
                smart_union=True,  # type: ignore
            ),
            validate_on_init=False,
            use_proxy=True,
        )

        return config_cls


if TYPE_CHECKING:
    # just a hack to prevent vscode from showing all the
    # mutable mapping methods for these dataclasses
    _MutableMappingBase = object
else:
    _MutableMappingBase = MutableMapping[str, Any]


def _to_json_default(obj: Any):
    # For types, we serialize them as a dict with the type name
    # and a special key to indicate that it is a type.
    if isinstance(obj, type):
        return {
            "__type__": True,
            "type": f"{obj.__module__}.{obj.__qualname__}",
        }

    return pydantic_encoder(obj)


def _from_json_object_hook(dct: Any):
    match dct:
        case {"__type__": True, "type": type}:
            # Split the typename into module and qualname
            split_type = type.rsplit(".", 1)
            # If there is no module, then we assume it is a builtin type
            if len(split_type) == 1:
                module_name = "builtins"
                qualname = split_type[0]
            elif len(split_type) == 2:
                module_name, qualname = split_type
            else:
                # This should never happen
                raise ValueError(f"Invalid type: {type}")

            # Import the module
            try:
                module = import_module(module_name)
            except ModuleNotFoundError as e:
                raise ValueError(
                    f"Failed to loaed {type=} from JSON because module {module_name} does not exist"
                ) from e

            try:
                return getattr(module, qualname)
            except AttributeError as e:
                raise ValueError(
                    f"Failed to loaed {type=} from JSON because {qualname} does not exist in {module_name}"
                ) from e
        case _:
            pass

    return dct


class TypedConfig(_MutableMappingBase, metaclass=TypedConfigMeta):
    # region construction methods
    @classmethod
    def default(cls):
        """
        Return a default instance of this class.
        For fields without default values, we explicitly set them to UNDEFINED.
        """

        return cls(
            **{
                field.name: UNDEFINED
                for field in fields(cls)
                if field.default is UNDEFINED and field.default_factory is UNDEFINED
            }
        )

    # endregion

    # region validation
    def validate(self):
        # When using pydantic w/ dataclasses, it will add a `__pydantic_validate_values__`
        # to the instance which we can use to delay the validation.
        getattr(self, "__pydantic_validate_values__")()

    # endregion

    # region JSON

    def json(self):
        return simplejson.dumps(
            self,
            default=_to_json_default,
            indent=4,
            sort_keys=True,
        )

    def yaml(self):
        # This is hacky, but we just convert the json to yaml
        return yaml.dump(simplejson.loads(self.json()))

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]):
        loaded = parse_obj_as(cls, source)
        return loaded

    @classmethod
    def from_json(cls, source: str):
        # Load the json as a dict
        loaded = simplejson.loads(source, object_hook=_from_json_object_hook)
        # Make sure it is a dict
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"The JSON representation of {cls} must be a dict, but got {type(loaded)}"
            )
        loaded = cls.from_dict(loaded)
        return loaded

    @classmethod
    def from_yaml(cls, source: str):
        # This is hacky, but we just convert the yaml to json and then load it
        # as json

        return cls.from_json(
            simplejson.dumps(
                yaml.safe_load(source),
                default=_to_json_default,
                indent=4,
                sort_keys=True,
            )
        )

    @classmethod
    def from_file(cls, path: str | Path):
        # Make sure we have a Path
        if not isinstance(path, Path):
            path = Path(path)

        # Load the config based on the file extension
        match path.suffix.lower():
            case ".yaml":
                config = cls.from_yaml(path.read_text())
                log.info(f"Loaded YAML config from {path}")
            case ".json":
                config = cls.from_json(path.read_text())
                log.info(f"Loaded JSON config from {path}")
            case _:
                raise ValueError(f"Unknown config file type: {path.suffix}")

        return config

    # endregion

    # region MutableMapping implementation
    # These are under `if not TYPE_CHECKING` to prevent vscode from showing
    # all the MutableMapping methods in the editor
    if not TYPE_CHECKING:

        @property
        def _ll_dict(self):
            return asdict(self)

        # we need to make sure every config class
        # is a MutableMapping[str, Any] so that it can be used
        # with lightning's hparams
        @override
        def __getitem__(self, key: str):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            value = self._ll_dict[first_key]

            for key in rest_keys:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)

            return value

        @override
        def __setitem__(self, key: str, value: Any):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            if len(rest_keys) == 0:
                self._ll_dict[first_key] = value
                return

            # we need to traverse the keys until we reach the last key
            # and then set the value
            current_value = self._ll_dict[first_key]
            for key in rest_keys[:-1]:
                if isinstance(current_value, Mapping):
                    current_value = current_value[key]
                else:
                    current_value = getattr(current_value, key)

            # set the value
            if isinstance(current_value, MutableMapping):
                current_value[rest_keys[-1]] = value
            else:
                setattr(current_value, rest_keys[-1], value)

        @override
        def __delitem__(self, key: str):
            # this is unsupported for this class
            raise NotImplementedError

        @override
        def __iter__(self):
            return iter(self._ll_dict)

        @override
        def __len__(self):
            return len(self._ll_dict)

    # endregion


__all__ = [
    "UNDEFINED",
    "Field",
    "TypedConfigMeta",
    "TypedConfig",
    "validator",
]
