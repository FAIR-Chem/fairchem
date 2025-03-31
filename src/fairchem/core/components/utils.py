"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fairchem.core.components.reducer import Reducer
    from fairchem.core.components.runner import Runner


class ManagedAttribute:
    """A descriptor helper to manage setting/access to an attribute of a class"""

    def __init__(self, enforced_type: type | None = None):
        self._enforced_type = enforced_type

    def __set_name__(self, owner: Runner | Reducer, name: str):
        self.public_name: str = name
        self.private_name: str = "_" + name

    def __get__(
        self, obj: Runner | Reducer, objtype: type[Runner | Reducer] | None = None
    ):
        return getattr(obj, self.private_name)

    def __set__(self, obj: Runner | Reducer, value: Any):
        if self._enforced_type is not None and not isinstance(
            value, self._enforced_type
        ):
            raise ValueError(
                f"{self.public_name} can only be set to an instance of {self._enforced_type} type!"
            )
        setattr(obj, self.private_name, value)
