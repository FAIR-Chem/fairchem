"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    import argparse

    from omegaconf import DictConfig
from submitit.helpers import Checkpointable, DelayedSubmission

from fairchem.core.common.flags import flags
from fairchem.core.common.utils import setup_imports
from fairchem.core.components.runner import Runner


class Submitit(Checkpointable):
    def __call__(self, dict_config) -> None:
        self.config = dict_config
        # TODO: this is not needed if we stop instantiating models with Registry.
        setup_imports()
        runner: Runner = hydra.utils.instantiate(dict_config.runner)
        runner.load_state()
        runner.run()

    def checkpoint(self, *args, **kwargs):
        logging.info("Submitit checkpointing callback is triggered")
        new_runner = Runner()
        new_runner.save_state()
        logging.info("Submitit checkpointing callback is completed")
        return DelayedSubmission(new_runner, self.config)


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory)
    # cfg = omegaconf.OmegaConf.load(args.config_yml)
    return hydra.compose(config_name=config_name, overrides=overrides_args)


# this is meant as a future replacement for the main entrypoint
def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config_yml, override_args)
    srunner = Submitit()
    logging.info("Running in local mode without elastic launch (single gpu only)")
    srunner(cfg)
