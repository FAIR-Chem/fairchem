from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile
from torchtnt.framework.callback import Callback

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torchtnt.framework import State, TTrainUnit

    from fairchem.core.common.logger import Logger


def get_default_profiler_handler(
    run_id: str, output_dir: str, logger: Logger, all_ranks: bool = False
):
    """Get a standard callback handle for the pytorch profiler"""

    def trace_handler(p):
        if all_ranks or distutils.is_master():
            trace_name = f"{run_id}_rank_{distutils.get_rank()}.pt.trace.json"
            output_path = os.path.join(output_dir, trace_name)
            logging.info(f"Saving trace in {output_path}")
            p.export_chrome_trace(output_path)
            if logger:
                logger.log_artifact(
                    name=trace_name, type="profile", file_location=output_path
                )

    return trace_handler


def get_profile_schedule(wait: int = 5, warmup: int = 5, active: int = 2):
    """Get a profile schedule and total number of steps to run
    check pytorch docs on the meaning of these paramters:
    https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule
    Example usage:
    ```
    trace_handler = get_default_profiler_handler(run_id = self.config["cmd"]["timestamp_id"],
                                                    output_dir = self.config["cmd"]["results_dir"],
                                                    logger = self.logger)
    profile_schedule, total_profile_steps = get_profile_schedule()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profile_schedule,
        on_trace_ready=trace_handler
    ) as p:
        for i in steps:
            <code block to profile>
            if i < total_profile_steps:
                p.step()
    """
    total_profile_steps = wait + warmup + active
    profile_schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)

    return profile_schedule, total_profile_steps


class ProfilerCallback(Callback):
    def __init__(
        self,
        job_config: DictConfig,
        wait_steps: int = 5,
        warmup_steps: int = 5,
        active_steps: int = 2,
        all_ranks: bool = False,
        activities: tuple = (ProfilerActivity.CPU, ProfilerActivity.CUDA),
    ) -> None:
        profile_dir = os.path.join(job_config.metadata.log_dir, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        logger = (
            WandBSingletonLogger.get_instance()
            if WandBSingletonLogger.initialized()
            else None
        )
        handler = get_default_profiler_handler(
            run_id=job_config.run_name,
            output_dir=profile_dir,
            logger=logger,
            all_ranks=all_ranks,
        )
        schedule, self.total_steps = get_profile_schedule(
            wait_steps, warmup_steps, active_steps
        )
        self.profiler = profile(
            activities=activities, schedule=schedule, on_trace_ready=handler
        )

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        if step <= self.total_steps:
            self.profiler.step()
        else:
            self.profiler.stop()
