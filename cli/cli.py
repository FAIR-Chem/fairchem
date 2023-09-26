"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar

import click
from click_option_group import optgroup

import ocpmodels
from cli.download_data import DOWNLOAD_LINKS_s2ef, get_data
from cli.runner import run_with_config
from ocpmodels.common.utils import build_config, setup_logging

T = TypeVar("T")


def _logging_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Common CLI options related to logging.
    """

    @optgroup.group("Logging")
    @optgroup.option(
        "--log-level",
        type=click.Choice(["ERROR", "WARNING", "WARN", "INFO", "DEBUG"]),
        default="INFO",
        envvar="LOG_LEVEL",
        help="The minimum level of log statements to print.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _runner_logging_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Extension of _logging_options that adds options common to all commands
    that use a Runner instance.
    """

    @_logging_options
    @optgroup.option(
        "--logdir",
        type=Path,
        default="logs",
        help="Name of the subdirectory where logs will be saved.",
    )
    @optgroup.option(
        "--timestamp-id",
        type=str,
        default=None,
        help="Overrides the ID used in timestamps.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _download_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Common CLI options related to downloading datasets.
    """

    @optgroup.group("Download")
    @optgroup.option(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Directory in which the dataset will be saved.",
    )
    @optgroup.option(
        "--keep",
        is_flag=True,
        default=False,
        help=(
            "Preserve intermediate directories and files after download "
            "and preprocessing steps have completed."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _runner_job_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Common CLI options to for job settings in Runner instances.
    """

    @optgroup.group("Job")
    @optgroup.option(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the model checkpoint to load.",
    )
    # TODO Add option to set checkpoint by name and download it
    @optgroup.option(
        "--config-yml",
        type=Path,
        required=True,
        help=(
            "Path to a config file that lists data, model, and optimization "
            "parameters."
        ),
    )
    @optgroup.option(
        "--config",
        type=str,
        multiple=True,
        default=[],
        help=(
            "Sets a single configuration option, taking precendence over "
            "values defined in the file pointed to by --config-yml. Use "
            "key=value with '.' characters to separate nested key names. "
            "For example: --config parent.config=val. Can be used more than "
            "once to set multiple key/value pairs."
        ),
    )
    @optgroup.option(
        "--sweep-yml",
        type=Path,
        default=None,
        help="Path to the config file with parameter sweeps.",
    )
    @optgroup.option(
        "--run-dir",
        type=str,
        default="./",
        help="Directory in which checkpoints, logs, and results will be saved.",
    )
    @optgroup.option(
        "--identifier",
        type=str,
        default="",
        help=(
            "Experimental identifier that will be appended to the name of the "
            "directory in which checkpoints, logs, and results are saved."
        ),
    )
    @optgroup.option(
        "--seed",
        type=int,
        default=0,
        help="Random seed for torch, cuda, and numpy.",
    )
    @optgroup.option(
        "--debug",
        is_flag=True,
        default=False,
        help="Run in debug mode.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _runner_cluster_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Common CLI options for cluster settings in Runner instances.
    """

    @optgroup.group("Cluster")
    @optgroup.option(
        "--submit",
        is_flag=True,
        default=False,
        help="Submit job to cluster.",
    )
    @optgroup.option(
        "--slurm-partition",
        type=str,
        default="ocp",
        help="Name of the slurm partition to submit to.",
    )
    @optgroup.option(
        "--slurm-mem",
        type=int,
        default=80,
        help="Memory limit, in GB, of jobs submitted with slurm.",
    )
    @optgroup.option(
        "--slurm-timeout",
        type=int,
        default=72,
        help="Time limit, in hours, of jobs submitted with slurm.",
    )
    @optgroup.option(
        "--summit",
        is_flag=True,
        default=False,
        help="Running on Summit cluster.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _runner_resource_options(func: Callable[..., T]) -> Callable[..., T]:
    """
    Common CLI options for resource settings in Runner instances.
    """

    @optgroup.group("Resources")
    @optgroup.option(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to request",
    )
    @optgroup.option(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to request.",
    )
    @optgroup.option(
        "--cpu",
        is_flag=True,
        default=False,
        help="Run with CPUs only.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def _runner_parallelization_options(
    func: Callable[..., T]
) -> Callable[..., T]:
    """
    Common CLI options for parallelization settings in Runner instances.
    """

    @optgroup.group("Parallelization")
    @optgroup.option(
        "--distributed",
        is_flag=True,
        default=False,
        help="Run with DDP (PyTorch Distributed Data Parallel).",
    )
    @optgroup.option(
        "--distributed-backend",
        type=str,
        default="nccl",
        help="Backend for DDP.",
    )
    @optgroup.option(
        "--distributed-port",
        type=int,
        default=13356,
        help="Port on master for DDP",
    )
    @optgroup.option(
        "--no-ddp",
        is_flag=True,
        default=False,
        help="Do not use DDP.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


@click.group(
    context_settings={"max_content_width": 120, "show_default": True},
    help="Graph networks for atomistic simulations",
)
def cli() -> None:
    pass


@cli.group(
    name="download",
    short_help="Download datasets",
    help="Download and preprocess datasets.",
)
def download():
    pass


@download.command(
    name="s2ef",
    short_help="Download s2ef datasets",
    help="Download and preprocess an s2ef dataset split.",
)
@click.argument(
    "split",
    type=click.Choice(DOWNLOAD_LINKS_s2ef["s2ef"].keys()),
    required=True,
)
@_download_options
@optgroup.group("Preprocess")
@optgroup.option(
    "--num-workers",
    type=int,
    default=1,
    help="Number of processes to launch when preprocessing data.",
)
@optgroup.option(
    "--get-edges",
    is_flag=True,
    default=False,
    help=(
        "Store edge indices in LMDB rather than compute them on the fly. This "
        "increases storage requirements by approximately 10x."
    ),
)
@optgroup.option(
    "--ref-energy",
    is_flag=True,
    default=False,
    help="Subtract reference energies.",
)
@optgroup.option(
    "--test-data",
    is_flag=True,
    default=False,
    help="Whether processing test data.",
)
@_logging_options
def download_s2ef(
    split: str,
    data_path: str,
    keep: bool,
    num_workers: int,
    get_edges: bool,
    ref_energy: bool,
    test_data: bool,
    log_level: str,
):
    setup_logging(level=log_level)
    get_data(
        datadir=data_path,
        task="s2ef",
        split=split,
        del_intmd_files=not keep,
        num_workers=num_workers,
        get_edges=get_edges,
        ref_energy=ref_energy,
        test_data=test_data,
    )


@download.command(
    name="is2re",
    short_help="Download is2re datasets",
    help="Download and preprocess is2re datasets.",
)
@_download_options
@_logging_options
def download_is2re(
    data_path: str,
    keep: bool,
    log_level: str,
):
    setup_logging(level=log_level)
    get_data(
        datadir=data_path,
        task="is2re",
        split=None,
        del_intmd_files=not keep,
    )


@cli.command(
    name="train",
    short_help="Train a new model",
    help="Train a new model.",
)
@_runner_job_options
@optgroup.option(
    "--amp",
    is_flag=True,
    default=False,
    help="Use mixed-precision training.",
)
@_runner_cluster_options
@_runner_resource_options
@_runner_parallelization_options
@optgroup.option(
    "--gp-gpus",
    type=int,
    default=None,
    help="Number of GPUs to split the graph over in Graph Parallel training.",
)
@optgroup.option(
    "--local-rank",
    type=int,
    default=0,
    help="Local rank in distributed training.",
)
@_runner_logging_options
@optgroup.option(
    "--print-every",
    type=int,
    default=10,
    help="Number of iterations to run before each log statement is generated.",
)
def train(
    checkpoint_path: Optional[str],
    config_yml: Path,
    config: Tuple[str, ...],
    sweep_yml: Optional[Path],
    run_dir: str,
    identifier: str,
    seed: int,
    debug: bool,
    amp: bool,
    submit: bool,
    slurm_partition: str,
    slurm_mem: int,
    slurm_timeout: int,
    summit: bool,
    num_nodes: int,
    num_gpus: int,
    cpu: bool,
    distributed: bool,
    distributed_backend: str,
    distributed_port: int,
    no_ddp: bool,
    gp_gpus: int,
    local_rank: int,
    log_level: str,
    logdir: str,
    timestamp_id: Optional[str],
    print_every: int,
):
    setup_logging(level=log_level)
    config = build_config(
        mode="train",
        config_yml=str(config_yml),
        config_overrides=list(config),
        identifier=identifier,
        timestamp_id=timestamp_id,
        seed=seed,
        debug=debug,
        run_dir=run_dir,
        print_every=print_every,
        amp=amp,
        checkpoint=checkpoint_path,
        cpu_only=cpu,
        submit=submit,
        summit=summit,
        local_rank=local_rank,
        distributed_port=distributed_port,
        distributed_backend=distributed_backend,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        no_ddp=no_ddp,
        gp_gpus=gp_gpus,
    )
    run_with_config(
        config=config,
        sweep_yml=sweep_yml,
        submit=submit,
        logdir=logdir,
        identifier=identifier,
        slurm_mem=slurm_mem,
        slurm_timeout=slurm_timeout,
        slurm_partition=slurm_partition,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        distributed=distributed,
    )


@cli.command(
    name="finetune",
    short_help="",
    help=(),
)
@_logging_options
def finetune(log_level: str):
    setup_logging(level=log_level)


@cli.command(
    name="validate",
    short_help="",
    help=(),
)
@_runner_job_options
@_runner_cluster_options
@_runner_resource_options
@_runner_parallelization_options
@_runner_logging_options
def validate(log_level: str):
    setup_logging(level=log_level)


@cli.command(
    name="predict",
    short_help="",
    help=(),
)
@_runner_job_options
@_runner_cluster_options
@_runner_resource_options
@_runner_parallelization_options
@_runner_logging_options
def predict(log_level: str):
    setup_logging(level=log_level)


@cli.command(
    name="relax",
    short_help="",
    help=(),
)
@_runner_job_options
@_runner_cluster_options
@_runner_resource_options
@_runner_parallelization_options
@_runner_logging_options
def relax(log_level: str):
    setup_logging(level=log_level)


if __name__ == "__main__":
    cli()
