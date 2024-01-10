"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import CLUSTER, JOB_ID

NTFY_OK = False
try:
    import ntfy_wrapper  # noqa: F401

    NTFY_OK = True
except ImportError:
    pass


class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, trainer_config):
        self.trainer_config = trainer_config
        self._ntfy = None
        self.url = None

    def collect_output_files(self):
        pass

    def finish(self, *args, **kwargs):
        pass

    def ntfy(self, *args, **kwargs):
        global NTFY_OK
        try:
            if NTFY_OK:
                if self._ntfy is None:
                    from ntfy_wrapper import Notifier

                    self._ntfy = Notifier(
                        notify_defaults={
                            "title": self.trainer_config.get("wandb_project", "OCP")
                        },
                        warnings=False,
                        verbose=False,
                    )
                self._ntfy(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Logger failed to send notification: {e}")
            NTFY_OK = False

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
        wandb_tags = note = name = None

        if trainer_config.get("wandb_resume_id"):
            wandb_id = trainer_config["wandb_resume_id"]
            print("⛑ Resuming wandb run: ", wandb_id)
        else:
            wandb_id = str(self.trainer_config.get("wandb_id", ""))
            if wandb_id:
                wandb_id += " - "
            if JOB_ID:
                wandb_id += f"{JOB_ID}-"
            wandb_id += self.trainer_config["config"]

            wandb_tags = trainer_config.get("wandb_tags", "")
            if wandb_tags:
                wandb_tags = [t.strip() for t in wandb_tags[:63].split(",")]
            note = self.trainer_config.get("note", "")
            name = self.trainer_config["wandb_name"] or wandb_id

        print("Initializing wandb run: ", wandb_id, "with name: ", name)

        self.run = wandb.init(
            config=self.trainer_config,
            id=wandb_id,
            name=name,
            dir=self.trainer_config["logs_dir"],
            project=self.trainer_config["wandb_project"],
            resume="allow",
            notes=note,
            tags=wandb_tags,
            entity="mila-ocp",
        )

        if "slurm_job_ids" not in self.run.config:
            self.run.config["slurm_job_ids"] = ""
        self.run.config.update(
            {
                "slurm_job_ids": ", ".join(
                    sorted(
                        set(
                            [
                                j.strip()
                                for j in self.run.config["slurm_job_ids"].split(",")
                            ]
                            + [JOB_ID]
                        )
                    )
                )
            },
            allow_val_change=True,
        )

        sbatch_files = list(
            Path(self.trainer_config["run_dir"]).glob("sbatch_script*.sh")
        )
        if len(sbatch_files) == 1 and not CLUSTER.drac:
            wandb.save(str(sbatch_files[0]))

        self.url = wandb.run.get_url()
        if self.url:
            with open(Path(self.trainer_config["run_dir"]) / "wandb_url.txt", "w") as f:
                f.write(self.url + "\n")
        if not CLUSTER.drac:
            self.collect_output_files(policy="live")
            self.collect_output_files(policy="end")
        print(f"\n{'-'*80}\n")

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

    def add_tags(self, tags):
        if not isinstance(tags, list):
            tags = [tags]
        tags = tuple(tags)
        self.run.tags = self.run.tags + tags

    def collect_output_files(self, policy="now"):
        outputs = Path(self.trainer_config["run_dir"]).glob("out*.txt")
        for o in outputs:
            wandb.save(str(o), policy=policy)

    def finish(self, error_or_signal=False):
        exit_code = 0
        if error_or_signal == "SIGTERM":
            self.add_tags("Preempted")
        if error_or_signal is True:
            exit_code = 1
        if not CLUSTER.drac:
            self.collect_output_files(policy="now")
        wandb.finish(exit_code=exit_code)


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

    def add_tags(self, tags):
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

    def add_tags(self, tags):
        pass

    def notify(self, *args, **kwargs):
        if not hasattr(self, "notifications"):
            self.notifications = []
        self.notifications.append((args, kwargs))
