from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any


class Runner(metaclass=ABCMeta):
    """
    Represents an abstraction over things that run in a loop and can save/load state.
    ie: Trainers, Validators, Relaxation all fall in this category.
    This allows us to decouple away from a monolithic trainer class
    """

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_state(self) -> None:
        raise NotImplementedError


# Used for testing
class MockRunner(Runner):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def run(self) -> Any:
        if self.x * self.y > 1000:
            raise ValueError("sum is greater than 1000!")
        return self.x + self.y

    def save_state(self) -> None:
        pass

    def load_state(self) -> None:
        pass
