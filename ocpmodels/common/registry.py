"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.
"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.

Import the global registry object using

``from ocpmodels.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a model: ``@registry.register_model``
"""
import importlib
from typing import Any, Callable, Dict, List, TypeVar, Union

R = TypeVar("R")
NestedDict = Dict[str, Union[str, Callable[..., Any], "NestedDict"]]


def _get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `ocpmodels.tasks.base_task.BaseTask`
    # we can use importlib to get the module (e.g., `ocpmodels.tasks.base_task`)
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"Could not import module `{module_name}` for import `{name}`"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class `{class_name}` from module `{module_name}`"
        ) from e


class Registry:
    r"""Class for registry object which acts as central source of truth."""
    mapping: NestedDict = {
        # Mappings to respective classes.
        "task_name_mapping": {},
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "logger_name_mapping": {},
        "trainer_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_task(cls, name: str):
        r"""Register a new task to registry with key 'name'
        Args:
            name: Key with which the task will be registered.
        Usage::
            from ocpmodels.common.registry import registry
            from ocpmodels.tasks import BaseTask
            @registry.register_task("train")
            class TrainTask(BaseTask):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["task_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls, name: str):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage::

            from ocpmodels.common.registry import registry
            from ocpmodels.datasets import BaseDataset

            @registry.register_dataset("qm9")
            class QM9(BaseDataset):
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name: str):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::

            from ocpmodels.common.registry import registry
            from ocpmodels.modules.layers import CGCNNConv

            @registry.register_model("cgcnn")
            class CGCNN():
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_logger(cls, name: str):
        r"""Register a logger to registry with key 'name'

        Args:
            name: Key with which the logger will be registered.

        Usage::

            from ocpmodels.common.registry import registry

            @registry.register_logger("wandb")
            class WandBLogger():
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            from ocpmodels.common.logger import Logger

            assert issubclass(
                func, Logger
            ), "All loggers must inherit Logger class"
            cls.mapping["logger_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_trainer(cls, name: str):
        r"""Register a trainer to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage::

            from ocpmodels.common.registry import registry

            @registry.register_trainer("active_discovery")
            class ActiveDiscoveryTrainer():
                ...
        """

        def wrap(func: Callable[..., R]) -> Callable[..., R]:
            cls.mapping["trainer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name: str, obj) -> None:
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from ocpmodels.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def __import_error(cls, name: str, mapping_name: str) -> RuntimeError:
        kind = mapping_name[: -len("_name_mapping")]
        mapping = cls.mapping.get(mapping_name, {})
        existing_keys: List[str] = list(mapping.keys())

        if len(existing_keys) == 0:
            raise RuntimeError(
                f"Registry for {mapping_name} is empty. You may have forgot to load a module."
            )

        existing_cls_path = (
            mapping.get(existing_keys[-1], None) if existing_keys else None
        )
        if existing_cls_path is not None:
            existing_cls_path = f"{existing_cls_path.__module__}.{existing_cls_path.__qualname__}"
        else:
            existing_cls_path = "ocpmodels.trainers.ForcesTrainer"

        existing_keys = [f"'{name}'" for name in existing_keys]
        existing_keys = (
            ", ".join(existing_keys[:-1]) + " or " + existing_keys[-1]
        )
        existing_keys_str = (
            f" (one of {existing_keys})" if existing_keys else ""
        )
        return RuntimeError(
            f"Failed to find the {kind} '{name}'. "
            f"You may either use a {kind} from the registry{existing_keys_str} "
            f"or provide the full import path to the {kind} (e.g., '{existing_cls_path}')."
        )

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}` (e.g., `ocpmodels.trainers.ForcesTrainer`)
        if name.count(".") < 1:
            raise cls.__import_error(name, mapping_name)

        try:
            return _get_absolute_mapping(name)
        except RuntimeError as e:
            raise cls.__import_error(name, mapping_name) from e

    @classmethod
    def get_task_class(cls, name: str):
        return cls.get_class(name, "task_name_mapping")

    @classmethod
    def get_dataset_class(cls, name: str):
        return cls.get_class(name, "dataset_name_mapping")

    @classmethod
    def get_model_class(cls, name: str):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_logger_class(cls, name: str):
        return cls.get_class(name, "logger_name_mapping")

    @classmethod
    def get_trainer_class(cls, name: str):
        return cls.get_class(name, "trainer_name_mapping")

    @classmethod
    def get(cls, name: str, default=None, no_warning: bool = False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from ocpmodels.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        split_name = name.split(".")
        value = cls.mapping["state"]
        for subname in split_name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name: str):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from ocpmodels.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
