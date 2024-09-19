from __future__ import annotations

from abc import ABCMeta, abstractmethod


class Runner(metaclass=ABCMeta):
    """
    Represents an abstraction over things that run in a loop and can save/load state.
    ie: Trainers, Validators, Relaxation all fall in this category.
    This allows us to decouple away from a monolithic trainer class
    """

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_state(self) -> None:
        raise NotImplementedError
