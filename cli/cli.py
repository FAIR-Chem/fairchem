"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import logging
import os
from typing import Any, Callable, TypeVar

import click
from click_option_group import optgroup

import ocpmodels
from cli.download_data import DOWNLOAD_LINKS_s2ef, get_data

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
    logging.basicConfig(level=log_level)
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
    logging.basicConfig(level=log_level)
    get_data(
        datadir=data_path,
        task="is2re",
        split=None,
        del_intmd_files=not keep,
    )


@cli.command(
    name="train",
    short_help="",
    help=(),
)
@_logging_options
def train(log_level: str):
    logging.basicConfig(level=log_level)


@cli.command(
    name="finetune",
    short_help="",
    help=(),
)
@_logging_options
def finetune(log_level: str):
    logging.basicConfig(level=log_level)


@cli.command(
    name="validate",
    short_help="",
    help=(),
)
@_logging_options
def validate(log_level: str):
    logging.basicConfig(level=log_level)


@cli.command(
    name="predict",
    short_help="",
    help=(),
)
@_logging_options
def predict(log_level: str):
    logging.basicConfig(level=log_level)


@cli.command(
    name="relax",
    short_help="",
    help=(),
)
@_logging_options
def relax(log_level: str):
    logging.basicConfig(level=log_level)


if __name__ == "__main__":
    cli()
