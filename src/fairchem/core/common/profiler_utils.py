from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from fairchem.core.common import distutils

if TYPE_CHECKING:
    from fairchem.core.common.logger import Logger

def get_default_profiler_handler(run_id: str, output_dir: str, logger: Logger):
    """Get a standard callback handle for the pytorch profiler"""

    def trace_handler(p):
        if distutils.is_master():
            trace_name = f"{run_id}_rank_{distutils.get_rank()}.pt.trace.json"
            output_path = os.path.join(output_dir, trace_name)
            print(f"Saving trace in {output_path}")
            p.export_chrome_trace(output_path)
            if logger:
                logger.log_artifact(name=trace_name, type="profile", file_location=output_path)
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
