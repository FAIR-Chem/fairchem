"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from fairchem.core.components.utils import ManagedAttribute

if TYPE_CHECKING:
    from fairchem.core.components.runner import Runner


class Reducer(metaclass=ABCMeta):
    """Represents an abstraction over things that reduce the results written by a set of runner.

    Note:
        When running with the `fairchemv2` cli, the `job_config` and `runner_config` attributes are set at
        runtime to those given in the config file.

    Attributes:
        job_config (DictConfig): a managed attribute that gives access to the job config
        runner_config (DictConfig): a managed attributed that gives access to the calling runner config
    """

    job_config = ManagedAttribute(enforced_type=DictConfig)
    runner_config = ManagedAttribute(enforced_type=DictConfig)

    @property
    @abstractmethod
    def runner_type(self) -> type[Runner]:
        """The runner type this reducer is associated with."""
        raise NotImplementedError

    @abstractmethod
    def reduce(self) -> Any:
        """Use file pattern to reduce"""
        raise NotImplementedError

    @abstractmethod
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, checkpoint_location: str | None) -> None:
        raise NotImplementedError


class MockReducer(Reducer):
    """Used for testing"""

    @property
    def runner_type(self):
        from fairchem.core.components.runner import MockRunner

        return MockRunner

    def reduce(self) -> Any:
        runner_path = self.runner_config.pop("_target_")
        print(
            f"Reducing results from runner {runner_path} with args: "
            f"{', '.join(f'{k}: {v}' for k, v in self.runner_config.items())}"
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
