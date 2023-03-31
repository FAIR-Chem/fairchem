"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import torch
from ocpmodels.common.registry import registry


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

    def run(self):
        try:
            loops = self.config.get("inference_time_loops", 5)
            if loops > 0:
                print("----------------------------------------")
                print("⏱️  Measuring inference time.")
                self.trainer.measure_inference_time(loops=loops)
                print("----------------------------------------\n")
            torch.set_grad_enabled(True)
            return self.trainer.train(
                disable_eval_tqdm=self.config.get("show_eval_progressbar", True),
                debug_batches=self.config.get("debug_batches", -1),
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e
