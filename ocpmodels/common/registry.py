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


class Registry:
    r"""Class for registry object which acts as central source of truth.
    """
    mapping = {
        # Mappings to respective classes.
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "logger_name_mapping": {},
        "trainer_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_dataset(cls, name):
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

        def wrap(func):
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
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

        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_logger(cls, name):
        r"""Register a logger to registry with key 'name'

        Args:
            name: Key with which the logger will be registered.

        Usage::

            from ocpmodels.common.registry import registry

            @registry.register_logger("tensorboard")
            class WandB():
                ...
        """

        def wrap(func):
            from ocpmodels.common.logger import Logger

            assert issubclass(
                func, Logger
            ), "All loggers must inherit Logger class"
            cls.mapping["logger_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_trainer(cls, name):
        r"""Register a trainer to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage::

            from ocpmodels.common.registry import registry

            @registry.register_trainer("active_discovery")
            class ActiveDiscoveryTrainer():
                ...
        """

        def wrap(func):
            cls.mapping["trainer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
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
    def get_dataset_class(cls, name):
        return cls.mapping["dataset_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_logger_class(cls, name):
        return cls.mapping["logger_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
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
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
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
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from ocpmodels.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
