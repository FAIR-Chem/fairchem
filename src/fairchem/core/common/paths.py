from __future__ import annotations

import datetime
import os

import torch

from fairchem.core.common import distutils

RESULT_DIR_NAME = "results"
CHECKPOINT_DIR_NAME = "checkpoints"
LOG_DIR_NAME = "logs"
DEFAULT_CHECKPOINT_NAME = "checkpoint.pt"

def unique_job_id(timestamp_id: str | None = None, slurm_job_id: str | None = None) -> str:
    # this is the standin for the unique job of the job, we use the slurm id if the slurm id exists
    # otherwise use the timestamp_id (uuid) of the job
    # at least 1 must exist
    assert not (timestamp_id is None and slurm_job_id is None)
    return timestamp_id if slurm_job_id is None else slurm_job_id


def get_result_dir(unique_job_id: str, run_dir: str) -> str:
    return os.path.join(run_dir,
                        unique_job_id,
                        RESULT_DIR_NAME)


def get_checkpoint_dir(unique_job_id: str, run_dir) -> str:
    return os.path.join(run_dir,
                        unique_job_id,
                        CHECKPOINT_DIR_NAME)


def get_log_dir(unique_job_id: str, run_dir: str) -> str:
    return os.path.join(run_dir,
                        unique_job_id,
                        LOG_DIR_NAME)

