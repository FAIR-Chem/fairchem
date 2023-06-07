from pathlib import Path

from lightning.pytorch import LightningDataModule, LightningModule


class LogDirMixin:
    @property
    def log_dir(self):
        if not isinstance(self, (LightningModule, LightningDataModule)):
            raise TypeError(
                "log_dir can only be used on LightningModule or LightningDataModule"
            )

        if (trainer := self.trainer) is None:
            raise RuntimeError("trainer is not defined")

        if (logger := trainer.logger) is None:
            raise RuntimeError("trainer.logger is not defined")

        if (log_dir := logger.log_dir) is None:
            raise RuntimeError("trainer.logger.log_dir is not defined")

        return Path(log_dir)
