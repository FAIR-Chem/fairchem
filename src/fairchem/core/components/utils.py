"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from fairchem.core.components.reducer import Reducer
    from fairchem.core.components.runner import Runner


class DictConfigAccess:
    """A descriptor helper to manage setting/access to a DictConfig attribute of a class"""

    def __set_name__(self, owner: Runner | Reducer, name: str):
        self.public_name: str = name
        self.private_name: str = "_" + name

    def __get__(
        self, obj: Runner | Reducer, objtype: type[Runner | Reducer] | None = None
    ):
        return getattr(obj, self.private_name)

    def __set__(self, obj: Runner | Reducer, value: DictConfig):
        if not isinstance(value, DictConfig):
            raise ValueError(
                f"{self.public_name} can only be set to an instance of {type(DictConfig)} type!"
            )
        setattr(obj, self.private_name, value)
