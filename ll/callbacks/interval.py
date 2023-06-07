from typing import Callable, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

Split = Literal["train", "val", "test", "predict"]


def _check_step(step: int, interval: int, skip_first: bool = False):
    if step % interval != 0:
        return False
    if skip_first and step == 0:
        return False
    return True


class StepIntervalCallback(Callback):
    def __init__(
        self,
        function: Callable[[Trainer, LightningModule], None],
        *,
        interval: int,
        skip_first: bool = False,
        splits: list[Split] = ["train", "val", "test", "predict"],
    ):
        super().__init__()

        self.function = function
        self.interval = interval
        self.skip_first = skip_first
        self.splits = set(splits)

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            not _check_step(
                trainer.global_step,
                self.interval,
                skip_first=self.skip_first,
            )
            or "train" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            not _check_step(
                trainer.global_step,
                self.interval,
                skip_first=self.skip_first,
            )
            or "val" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            not _check_step(
                trainer.global_step,
                self.interval,
                skip_first=self.skip_first,
            )
            or "test" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            not _check_step(
                trainer.global_step,
                self.interval,
                skip_first=self.skip_first,
            )
            or "predict" not in self.splits
        ):
            return
        self.function(trainer, pl_module)


class EpochIntervalCallback(Callback):
    def __init__(
        self,
        function: Callable[[Trainer, LightningModule], None],
        *,
        interval: int,
        skip_first: bool = False,
        splits: list[Split] = ["train", "val", "test", "predict"],
    ):
        super().__init__()

        self.function = function
        self.interval = interval
        self.skip_first = skip_first
        self.splits = set(splits)

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        if (
            not _check_step(
                trainer.current_epoch,
                self.interval,
                skip_first=self.skip_first,
            )
            or "train" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_validation_epoch_start(self, trainer, pl_module):
        if (
            not _check_step(
                trainer.current_epoch,
                self.interval,
                skip_first=self.skip_first,
            )
            or "val" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_test_epoch_start(self, trainer, pl_module):
        if (
            not _check_step(
                trainer.current_epoch,
                self.interval,
                skip_first=self.skip_first,
            )
            or "test" not in self.splits
        ):
            return
        self.function(trainer, pl_module)

    @override
    def on_predict_epoch_start(self, trainer, pl_module):
        if (
            not _check_step(
                trainer.current_epoch,
                self.interval,
                skip_first=self.skip_first,
            )
            or "predict" not in self.splits
        ):
            return
        self.function(trainer, pl_module)


class IntervalCallback(Callback):
    def __init__(
        self,
        function: Callable[[Trainer, LightningModule], None],
        *,
        step_interval: int | None = None,
        epoch_interval: int | None = None,
        skip_first: bool = False,
        splits: list[Split] = ["train", "val", "test", "predict"],
    ):
        super().__init__()

        self.callback = None

        if step_interval is not None:
            self.callback = StepIntervalCallback(
                function,
                interval=step_interval,
                splits=splits,
                skip_first=skip_first,
            )
        elif epoch_interval is not None:
            self.callback = EpochIntervalCallback(
                function,
                interval=epoch_interval,
                splits=splits,
                skip_first=skip_first,
            )
        else:
            raise ValueError("Either step_interval or epoch_interval must be specified")

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not isinstance(self.callback, StepIntervalCallback):
            return

        self.callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    @override
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not isinstance(self.callback, StepIntervalCallback):
            return

        self.callback.on_validation_batch_start(trainer, pl_module, batch, batch_idx)

    @override
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not isinstance(self.callback, StepIntervalCallback):
            return

        self.callback.on_test_batch_start(trainer, pl_module, batch, batch_idx)

    @override
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not isinstance(self.callback, StepIntervalCallback):
            return

        self.callback.on_predict_batch_start(trainer, pl_module, batch, batch_idx)

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        if not isinstance(self.callback, EpochIntervalCallback):
            return

        self.callback.on_train_epoch_start(trainer, pl_module)

    @override
    def on_validation_epoch_start(self, trainer, pl_module):
        if not isinstance(self.callback, EpochIntervalCallback):
            return

        self.callback.on_validation_epoch_start(trainer, pl_module)

    @override
    def on_test_epoch_start(self, trainer, pl_module):
        if not isinstance(self.callback, EpochIntervalCallback):
            return

        self.callback.on_test_epoch_start(trainer, pl_module)

    @override
    def on_predict_epoch_start(self, trainer, pl_module):
        if not isinstance(self.callback, EpochIntervalCallback):
            return

        self.callback.on_predict_epoch_start(trainer, pl_module)
