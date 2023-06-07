from typing import Any

from typing_extensions import override


class SkipBatch(BaseException):
    """Exception to skip the current batch."""


class TrainingError(Exception):
    """Exception thrown during training which contains information about the batch that caused the error.

    Args:
        exception: The exception that was thrown.
        batch_idx: The index of the batch that caused the error.
        batch: The batch that caused the error.
        epoch: The epoch that the error occurred in.
        global_step: The global step that the error occurred in.
        training_fn: The training function that the error occurred in (one of "training_fn", "validation_step", "test_step", "predict_step").
    """

    @override
    def __init__(
        self,
        exception: BaseException,
        *,
        batch_idx: int,
        batch: Any,
        epoch: int,
        global_step: int,
        training_fn: str,
    ):
        self.exception = exception
        self.batch_idx = batch_idx
        self.batch = batch
        self.epoch = epoch
        self.global_step = global_step
        self.training_fn = training_fn

        super().__init__(
            f"Training error in training_fn {training_fn} at epoch {epoch} and global step {global_step} at batch {batch_idx}."
        )

    @override
    def __repr__(self):
        return f"TrainingError(batch_idx={self.batch_idx}, batch={self.batch}, epoch={self.epoch}, global_step={self.global_step}, training_fn={self.training_fn})"

    @override
    def __str__(self):
        return self.__repr__()
