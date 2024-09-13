from __future__ import annotations

import argparse

from fairchem.core.models.equiformer_v2.eqv2_to_eqv2_hydra import (
    convert_checkpoint_and_config_to_hydra,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eqv2-checkpoint", help="path to eqv2 checkpoint", type=str, required=True
    )
    parser.add_argument(
        "--eqv2-yaml", help="path to eqv2 yaml config", type=str, required=True
    )
    parser.add_argument(
        "--hydra-eqv2-checkpoint",
        help="path where to output hydra checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hydra-eqv2-yaml",
        help="path where to output hydra yaml",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    convert_checkpoint_and_config_to_hydra(
        yaml_fn=args.eqv2_yaml,
        checkpoint_fn=args.eqv2_checkpoint,
        new_yaml_fn=args.hydra_eqv2_yaml,
        new_checkpoint_fn=args.hydra_eqv2_checkpoint,
    )
