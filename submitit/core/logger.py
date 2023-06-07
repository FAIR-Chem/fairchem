# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging.config
import os
from typing import Union

# provide a way to change level through SUBMITIT_LOG_LEVEL environment variable:
# level "CRITICAL" (50) or more (eg.: "100") will deactivate submitit logger
# "NOCONFIG" will avoid configuration
LOG_VARNAME = "SUBMITIT_LOG_LEVEL"
level_str = os.environ.get(LOG_VARNAME, "INFO").upper()
level: Union[int, str] = level_str if not level_str.isdigit() else int(level_str)


CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"submitit_basic": {"format": "%(name)s %(levelname)s (%(asctime)s) - %(message)s"}},
    "handlers": {
        "submitit_out": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "submitit_basic",
            "stream": "ext://sys.stdout",
        },
        "submitit_err": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "submitit_basic",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {"submitit": {"handlers": ["submitit_err", "submitit_out"], "level": level}},
}


if level != "NOCONFIG":
    logging.config.dictConfig(CONFIG)


def get_logger() -> logging.Logger:
    return logging.getLogger("submitit")


def exception(*args: str) -> None:
    get_logger().exception(*args)


def warning(*args: str) -> None:
    get_logger().warning(*args)
