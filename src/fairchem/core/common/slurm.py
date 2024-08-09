from __future__ import annotations

import os


def update_slurm_config(slurm_config: dict) -> dict:
    if "SLURM_JOB_ID" in os.environ and "folder" in slurm_config:
        if "SLURM_ARRAY_JOB_ID" in os.environ:
            slurm_config["job_id"] = "{}_{}".format(
                os.environ["SLURM_ARRAY_JOB_ID"],
                os.environ["SLURM_ARRAY_TASK_ID"],
            )
        else:
            slurm_config["job_id"] = os.environ["SLURM_JOB_ID"]
        slurm_config["folder"] = slurm_config["folder"].replace(
            "%j", slurm_config["job_id"]
        )
    return slurm_config
