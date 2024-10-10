from __future__ import annotations

import sys

import hydra
import pytest

from fairchem.core._cli import main
from fairchem.core.common import distutils


def test_hydra_cli():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = ["--hydra", "--config-yml", "tests/core/test_hydra_cli.yml", "--cpu"]
    sys.argv[1:] = sys_args
    main()


def test_hydra_cli_throws_error():
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys_args = [
        "--hydra",
        "--cpu",
        "--config-yml",
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
        "--hydra",
        "--cpu",
        "--config-yml",
        "tests/core/test_hydra_cli.yml",
        "runner.x=1000",
        "runner.z=5",  # z is not a valid input argument to runner
    ]
    sys.argv[1:] = sys_args
    with pytest.raises(hydra.errors.ConfigCompositionException):
        main()
