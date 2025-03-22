from __future__ import annotations

import sys

import hydra
import pytest

from fairchem.core._cli_hydra import (
    ALLOWED_TOP_LEVEL_KEYS,
    get_hydra_config_from_yaml,
    main,
)
from fairchem.core.common import distutils


def test_hydra_cli():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", "tests/core/test_hydra_cli.yml"]
    sys.argv[1:] = sys_args
    main()


def test_hydra_cli_run_reduce():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--config", "tests/core/test_hydra_cli_run_reduce.yml"]
    sys.argv[1:] = sys_args
    main()


def test_hydra_cli_throws_error():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "--config",
        "tests/core/test_hydra_cli.yml",
        "runner.x=1000",
        "runner.y=5",
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(ValueError) as error_info:
        main()
    assert "sum is greater than 1000" in str(error_info.value)


def test_hydra_cli_throws_error_on_invalid_inputs():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "-c",
        "tests/core/test_hydra_cli.yml",
        "runner.x=1000",
        "runner.a=5",  # a is not a valid input argument to runner
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(hydra.errors.ConfigCompositionException):
        main()


def test_hydra_cli_throws_error_on_disallowed_top_level_keys():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    assert "x" not in ALLOWED_TOP_LEVEL_KEYS
    sys_args = [
        "-c",
        "tests/core/test_hydra_cli.yml",
        "+x=1000",  # this is not allowed because we are adding a key that is not in ALLOWED_TOP_LEVEL_KEYS
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(ValueError):
        main()


def get_cfg_from_yaml():
    yaml = "tests/core/test_hydra_cli.yml"
    cfg = get_hydra_config_from_yaml(yaml)
    # assert fields got initialized properly
    assert cfg.job.run_name is not None
    assert cfg.job.seed is not None
    assert cfg.keys() == ALLOWED_TOP_LEVEL_KEYS
