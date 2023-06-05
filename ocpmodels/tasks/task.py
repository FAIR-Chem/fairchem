"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from pathlib import Path

import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import set_deup_samples_path


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config.get("checkpoint") is not None:
            print("\n🔵 Resuming:\n  • ", end="", flush=True)
            self.trainer.load_checkpoint(self.config["checkpoint"])
            print()

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. "
                        + "Consider removing it from the model."
                    )

    def create_deup_dataset(self):
        dds = self.config["deup_dataset"]
        if self.config["trainer"] != "deup":
            ensemble_trainer = registry.get_trainer_class("deup")(
                "deup_faenet-deup-all",
                trainers_conf={
                    "checkpoints": (
                        Path(self.config["checkpoint_dir"]) / "best_checkpoint.pt"
                    ),
                    "dropout": self.config["model"].get("dropout_lin") or 0.75,
                },
                overrides={"logger": "dummy"},
            )
        else:
            ensemble_trainer = self.trainer
        output_path = ensemble_trainer.create_deup_dataset(
            dds["dataset_strs"],
            dds["n_samples"],
            dds.get("output_path") or Path(self.config["run_dir"]) / "deup_dataset",
            -1,
        )
        print("\n🤠 DEUP Dataset created in:", str(output_path))
        return output_path

    def run(self):
        self.config = self.trainer.config
        try:
            if self.config.get("deup_dataset", {}).get("create") == "before":
                output_path = self.create_deup_dataset()
                # self.trainer must be an EnsembleTrainer at this point
                self.trainer.config["deup_samples_path"] = str(output_path)
                self.trainer.config = set_deup_samples_path(self.trainer.config)
                self.trainer.load()

            loops = self.config.get("inference_time_loops", 5)
            if loops > 0:
                print("----------------------------------------")
                print("⏱️  Measuring inference time.")
                self.trainer.measure_inference_time(loops=loops)
                print("----------------------------------------\n")
            torch.set_grad_enabled(True)
            training_signal = self.trainer.train(
                disable_eval_tqdm=self.config.get("show_eval_progressbar", True),
                debug_batches=self.config.get("debug_batches", -1),
            )
            if training_signal == "SIGTERM":
                return

            if self.config.get("deup_dataset", {}).get("create") == "after":
                self.create_deup_dataset()

        except RuntimeError as e:
            self._process_error(e)
            raise e
