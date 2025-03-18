"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Reducer(metaclass=ABCMeta):
    """
    Represents an abstraction over things reduce the results written by a runner.
    """

    file_pattern: ClassVar[str] = "*"

    def run(self) -> Any:
        self.reduce()

    @abstractmethod
    def initialize(self, job_config: DictConfig, runner_config: DictConfig) -> None:
        """Initialize takes both the job config and a runner config assumed to have been run beforehand"""
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
