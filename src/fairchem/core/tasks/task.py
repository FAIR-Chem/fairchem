"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import get_checkpoint_format
from fairchem.core.trainers import OCPTrainer


class BaseTask:
    def __init__(self, config) -> None:
        self.config = config

    def setup(self, trainer) -> None:
        self.trainer = trainer

        format = get_checkpoint_format(self.config)
        if format == "pt":
            self.chkpt_path = os.path.join(
                self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
            )
        else:
            self.chkpt_path = self.trainer.config["cmd"]["checkpoint_dir"]

        # if the supplied checkpoint exists, then load that, ie: when user specifies the --checkpoint option
        # OR if the a job was preempted correctly and the submitit checkpoint function was called
        # (https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py#L44), then we should attempt to
        # load that checkpoint
        if self.config["checkpoint"] is not None:
            logging.info(
                f"Attemping to load user specified checkpoint at {self.config['checkpoint']}"
            )
            self.trainer.load_checkpoint(checkpoint_path=self.config["checkpoint"])
        # if the supplied checkpoint doesn't exist and there exists a previous checkpoint in the checkpoint path, this
        # means that the previous job didn't terminate "nicely" (due to node failures, crashes etc), then attempt
        # to load the last found checkpoint
        elif (
            os.path.isfile(self.chkpt_path)
            or (os.path.isdir(self.chkpt_path) and len(os.listdir(self.chkpt_path))) > 0
        ):
            logging.info(
                f"Previous checkpoint found at {self.chkpt_path}, resuming job from this checkecpoint"
            )
            self.trainer.load_checkpoint(checkpoint_path=self.chkpt_path)

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError) -> None:
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self) -> None:
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self) -> None:
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self) -> None:
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("run-relaxations")
class RelaxationTask(BaseTask):
    def run(self) -> None:
        assert isinstance(
            self.trainer, OCPTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()
