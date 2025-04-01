"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

from omegaconf import DictConfig

from fairchem.core.components.utils import ManagedAttribute


class Runner(metaclass=ABCMeta):
    """Represents an abstraction over things that run in a loop and can save/load state.

    ie: Trainers, Validators, Relaxation all fall in this category.

    Note:
        When running with the `fairchemv2` cli, the `job_config` and attribute is set at
        runtime to those given in the config file.

    Attributes:
        job_config (DictConfig): a managed attribute that gives access to the job config
    """

    job_config = ManagedAttribute(enforced_type=DictConfig)

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, checkpoint_location: str | None) -> None:
        raise NotImplementedError


class MockRunner(Runner):
    """Used for testing"""

    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def run(self) -> Any:
        if self.x + self.y > 1000:
            raise ValueError("sum is greater than 1000!")
        return self.x + self.y

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
