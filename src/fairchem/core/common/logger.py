"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import tensor_stats


class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def watch(self, model, log_freq: int = 1000):
        """
        Monitor parameters and gradients.
        """

    def log(self, update_dict, step: int, split: str = ""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict
        return update_dict

    @abstractmethod
    def log_plots(self, plots) -> None:
        pass

    @abstractmethod
    def mark_preempting(self) -> None:
        pass

    @abstractmethod
    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        pass


@registry.register_logger("wandb")
class WandBLogger(Logger):
    def __init__(self, config) -> None:
        super().__init__(config)
        project = (
            self.config["logger"].get("project", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        entity = (
            self.config["logger"].get("entity", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        group = (
            self.config["logger"].get("group", None)
            if isinstance(self.config["logger"], dict)
            else None
        )

        wandb.init(
            config=self.config,
            id=self.config["cmd"]["timestamp_id"],
            name=self.config["cmd"]["identifier"],
            dir=self.config["cmd"]["logs_dir"],
            project=project,
            entity=entity,
            resume="allow",
            group=group,
        )

    def watch(self, model, log="all", log_freq: int = 1000) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, update_dict, step: int, split: str = "") -> None:
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption: str = "") -> None:
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_table(
        self, name: str, cols: list, data: list, step: int | None = None, commit=False
    ) -> None:
        # cols are 1D list of N elements, data must be NxK where the number of cols must match cols
        # see https://docs.wandb.ai/guides/tables
        table = wandb.Table(columns=cols, data=data)
        wandb.log({name: table}, step=step, commit=commit)

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        art = wandb.Artifact(name=name, type=type)
        art.add_file(file_location)
        art.save()


@registry.register_logger("tensorboard")
class TensorboardLogger(Logger):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.writer = SummaryWriter(self.config["cmd"]["logs_dir"])

    # TODO: add a model hook for watching gradients.
    def watch(self, model, log_freq: int = 1000) -> bool:
        logging.warning("Model gradient logging to tensorboard not yet supported.")
        return False

    def log(self, update_dict, step: int, split: str = ""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], (int, float))
                self.writer.add_scalar(key, update_dict[key], step)

    def mark_preempting(self) -> None:
        logging.warning("mark_preempting for Tensorboard not supported")

    def log_plots(self, plots) -> None:
        logging.warning("log_plots for Tensorboard not supported")

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        logging.warning("log_summary for Tensorboard not supported")

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        logging.warning("log_artifact for Tensorboard not supported")


class WandBSingletonLogger:
    """
    Singleton version of wandb logger, this forces a single instance of the logger to be created and used from anywhere in the code (not just from the trainer).
    This will replace the original WandBLogger.

    We initialize wandb instance somewhere in the trainer/runner globally:

    WandBSingletonLogger.init_wandb(...)

    Then from anywhere in the code we can fetch the singleton instance and log to wandb,
    note this allows you to log without knowing explicitly which step you are on
    see: https://docs.wandb.ai/ref/python/log/#the-wb-step for more details

    WandBSingletonLogger.get_instance().log({"some_value": value}, commit=False)
    """

    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def initialized(cls) -> bool:
        return WandBSingletonLogger._instance is not None

    @classmethod
    def init_wandb(
        cls,
        config: dict,
        run_id: str,
        run_name: str,
        log_dir: str,
        project: str,
        entity: str,
        group: str | None = None,
    ) -> None:
        wandb.init(
            config=config,
            id=run_id,
            name=run_name,
            dir=log_dir,
            project=project,
            entity=entity,
            resume="allow",
            group=group,
        )

    @classmethod
    def get_instance(cls):
        assert wandb.run is not None, "wandb is not initialized, call init_wandb first!"
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def watch(self, model, log="all", log_freq: int = 1000) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(
        self, update_dict: dict, step: int | None = None, commit=False, split: str = ""
    ) -> None:
        # HACK: this is really ugly logic here for backward compat but we should get rid of.
        # the split string shouldn't inserted here
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict

        # if step is not specified, wandb will use an auto-incremented step: https://docs.wandb.ai/ref/python/log/
        # otherwise the user must increment it manually (not recommended)
        wandb.log(update_dict, step=step, commit=commit)

    def log_table(
        self, name: str, cols: list, data: list, step: int | None = None, commit=False
    ) -> None:
        # cols are 1D list of N elements, data must be NxK where the number of cols must match cols
        # see https://docs.wandb.ai/guides/tables
        table = wandb.Table(columns=cols, data=data)
        wandb.log({name: table}, step=step, commit=commit)

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        art = wandb.Artifact(name=name, type=type)
        art.add_file(file_location)
        art.save()


# convienience function for logging stats with WandBSingletonLogger
def log_stats(x: torch.Tensor, prefix: str):
    if distutils.is_master() and WandBSingletonLogger._instance is not None:
        WandBSingletonLogger.get_instance().log(
            tensor_stats(prefix, x),
            commit=False,
        )
