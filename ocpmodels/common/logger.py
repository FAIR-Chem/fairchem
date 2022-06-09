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

    def __init__(self, config):
        self.config = config

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
    def __init__(self, config):
        super().__init__(config)
        project = (
            self.config["logger"].get("project", None)
            if isinstance(self.config["logger"], dict)
            else None
        )

        wandb_id = ""
        slurm_jobid = os.environ.get("SLURM_JOB_ID")
        if slurm_jobid:
            wandb_id += f"{slurm_jobid}-"
        wandb_id += self.config["cmd"]["timestamp_id"] + "-" + config["model"]

        wandb.init(
            config=self.config,
            id=wandb_id,
            name=self.config["cmd"]["identifier"],
            dir=self.config["cmd"]["logs_dir"],
            project=project,
            resume="allow",
            notes=self.config["note"],
        )

        sbatch_files = list(Path(self.config["run_dir"]).glob("sbatch_script*.sh"))
        if len(sbatch_files) == 1:
            wandb.save(str(sbatch_files[0]))

        with open(Path(self.config["run_dir"] / "wandb_url.txt"), "w") as f:
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
    def __init__(self, config):
        super().__init__(config)
        self.writer = SummaryWriter(self.config["cmd"]["logs_dir"])

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
