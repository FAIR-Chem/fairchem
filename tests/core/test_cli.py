from __future__ import annotations

import sys
from unittest.mock import patch

from fairchem.core._cli import main


def fake_runner(config: dict):
    assert config["world_size"] == 1


def test_cli():
    with patch("fairchem.core._cli.runner_wrapper", fake_runner):
        sys_args = [
            "--debug",
            "--mode",
            "train",
            "--identifier",
            "test",
            "--config-yml",
            "configs/oc22/s2ef/equiformer_v2/equiformer_v2_N@18_L@6_M@2_e4_f100_121M.yml",
        ]
        sys.argv[1:] = sys_args
        main()


def test_cli_multi_rank():
    with patch("fairchem.core._cli.elastic_launch") as mock_elastic_launch:
        sys_args = [
            "--debug",
            "--mode",
            "train",
            "--identifier",
            "test",
            "--config-yml",
            "configs/oc22/s2ef/equiformer_v2/equiformer_v2_N@18_L@6_M@2_e4_f100_121M.yml",
            "--cpu",
            "--num-gpus",
            "2",
        ]
        sys.argv[1:] = sys_args
        main()
        mock_elastic_launch.assert_called_once()
