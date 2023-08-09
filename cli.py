"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import logging
from typing import Callable, TypeVar

import click
from click_option_group import optgroup

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
    def wrapper(*args, **kwargs) -> T:
        return func(*args, **kwargs)

    return wrapper


@click.group(
    context_settings={"max_content_width": 120, "show_default": True},
    help="Graph networks for electrocatalyst design",
)
def cli() -> None:
    pass


@cli.command(
    name="download",
    short_help="",
    help=(),
)
@_logging_options
def download(log_level: str):
    logging.basicConfig(level=log_level)


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
