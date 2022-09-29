"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from ocpmodels.common.registry import registry


class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, trainer_config):
        self.trainer_config = trainer_config

    @abstractmethod
    def watch(self, model):
        """
        Monitor parameters and gradients.
        """
        pass

    def log(self, update_dict, step=None, split=""):
        """
        Log some values.
        """
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        return update_dict

    @abstractmethod
    def log_plots(self, plots):
        pass

    @abstractmethod
    def mark_preempting(self):
        pass


@registry.register_logger("wandb")
class WandBLogger(Logger):
    def __init__(self, trainer_config):
        super().__init__(trainer_config)
        project = (
            self.trainer_config["logger"].get("project", None)
            if isinstance(self.trainer_config["logger"], dict)
            else None
        )

        wandb_id = ""
        slurm_jobid = os.environ.get("SLURM_JOB_ID")
        if slurm_jobid:
            wandb_id += f"{slurm_jobid}-"
        wandb_id += (
            self.trainer_config["timestamp_id"] + "-" + trainer_config["model_name"]
        )

        wandb_tags = [
            t.strip() for t in trainer_config.get("wandb_tags", "").split(",")
        ]

        wandb.init(
            config=self.trainer_config,
            id=wandb_id,
            name=self.trainer_config["wandb_name"] or wandb_id,
            dir=self.trainer_config["logs_dir"],
            project=project,
            resume="allow",
            notes=self.trainer_config["note"],
            tags=wandb_tags,
        )

        sbatch_files = list(
            Path(self.trainer_config["run_dir"]).glob("sbatch_script*.sh")
        )
        if len(sbatch_files) == 1:
            wandb.save(str(sbatch_files[0]))

        with open(Path(self.trainer_config["run_dir"] / "wandb_url.txt"), "w") as f:
            f.write(wandb.run.get_url())

    def watch(self, model):
        wandb.watch(model)

    def log(self, update_dict, step=None, split=""):
        if step is not None:
            update_dict = super().log(update_dict, step, split)
            wandb.log(update_dict, step=int(step))
        else:
            update_dict = super().log(update_dict, split=split)
            wandb.log(update_dict)

    def log_plots(self, plots, caption=""):
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def mark_preempting(self):
        wandb.mark_preempting()


@registry.register_logger("tensorboard")
class TensorboardLogger(Logger):
    def __init__(self, trainer_config):
        super().__init__(trainer_config)
        self.writer = SummaryWriter(self.trainer_config["logs_dir"])

    # TODO: add a model hook for watching gradients.
    def watch(self, model):
        logging.warning("Model gradient logging to tensorboard not yet supported.")
        return False

    def log(self, update_dict, step=None, split=""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], int) or isinstance(
                    update_dict[key], float
                )
                self.writer.add_scalar(key, update_dict[key], step)

    def mark_preempting(self):
        pass

    def log_plots(self, plots):
        pass


@registry.register_logger("dummy")
class DummyLogger(Logger):
    def __init__(self, trainer_config):
        super().__init__(trainer_config)

    def log_plots(self, plots):
        pass

    def mark_preempting(self):
        pass

    def watch(self, model):
        pass
