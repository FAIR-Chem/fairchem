import os
from contextlib import contextmanager
from logging import getLogger

log = getLogger(__name__)


@contextmanager
def remove_slurm_environment_variables():
    """
    SLURM_CPU_BIND_* environment variables are set by SLURM in the current environment.
    We need to remove all of these environment variables during the codepath in which we create the new SLURM runs, so that the new SLURM runs do not inherit the environment variables from the current environment.
    To make things easier, we will patch the environment to remove all "SLURM_" environment variables.
    Otherwise, the runs will faill with an error like shown below:
        srun: error: CPU binding outside of job step allocation, allocated CPUs are: 0x01F000000001F0000000.
        srun: error: Task launch for StepId=5216715.0 failed on node learnfair0537: Unable to satisfy cpu bind request
        srun: error: Application launch failed: Unable to satisfy cpu bind request
        srun: Job step aborted

    See https://www.mail-archive.com/slurm-users@lists.schedmd.com/msg09157.html for more details.
    """

    removed_env_vars = {}
    for key in list(os.environ.keys()):
        if not key.startswith("SLURM_"):
            continue
        removed_env_vars[key] = os.environ.pop(key)

    log.debug(
        f"Removed environment variables before launching new SLURM job: {list(removed_env_vars.keys())}"
    )
    try:
        yield
    finally:
        os.environ.update(removed_env_vars)
        log.debug(
            f"Restored environment variables after launching new SLURM job: {list(removed_env_vars.keys())}"
        )


@contextmanager
def remove_wandb_environment_variables():
    """
    Similar to above, but removes all "WANDB_" environment variables.
    """

    removed_env_vars = {}
    for key in list(os.environ.keys()):
        if not key.startswith("WANDB_"):
            continue
        removed_env_vars[key] = os.environ.pop(key)

    log.debug(
        f"Removed environment variables before launching new SLURM job: {list(removed_env_vars.keys())}"
    )
    try:
        yield
    finally:
        os.environ.update(removed_env_vars)
        log.debug(
            f"Restored environment variables after launching new SLURM job: {list(removed_env_vars.keys())}"
        )


@contextmanager
def set_additional_env_vars(additional_env_vars: dict[str, str] | None = None):
    """
    Set additional environment variables for the run.
    Newly set environment variables will be removed after the run is finished.
    Existing environment variables will be restored to their original values after the run is finished.
    """
    if additional_env_vars is None:
        additional_env_vars = {}

    removed_env_vars = {}
    for key, value in additional_env_vars.items():
        removed_env_vars[key] = os.environ.pop(key, None)
        os.environ[key] = value

    log.debug(
        f"Set additional environment variables for the run: {list(additional_env_vars.keys())}"
    )
    try:
        yield
    finally:
        for key in additional_env_vars.keys():
            if removed_env_vars[key] is None:
                del os.environ[key]
            else:
                os.environ[key] = removed_env_vars[key]
        log.debug(
            f"Restored environment variables after launching new SLURM job: {list(additional_env_vars.keys())}"
        )
