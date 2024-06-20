from unittest.mock import patch
import sys

from fairchem.core._cli import main

def fake_runner(distributed: bool, config: dict):
    assert not distributed
    assert config["local_rank"] == 0
    assert config["world_size"] == 1

def test_cli():
    with patch('fairchem.core._cli.runner_wrapper',fake_runner):
        sys_args = ['--debug', 
                    '--mode', 
                    'train', 
                    '--identifier', 
                    'test', 
                    '--config-yml', 
                    'configs/oc22/s2ef/equiformer_v2/equiformer_v2_N@18_L@6_M@2_e4_f100_121M.yml']
        sys.argv[1:] = sys_args
        main()