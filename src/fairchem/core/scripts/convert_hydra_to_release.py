from __future__ import annotations

import argparse
import logging

import torch
import yaml


def convert_fine_tune_checkpoint(
    fine_tune_checkpoint_fn,
    output_checkpoint_fn,
    fine_tune_yaml_fn=None,
    output_yaml_fn=None,
):
    fine_tune_checkpoint = torch.load(fine_tune_checkpoint_fn, map_location="cpu")

    if "config" not in fine_tune_checkpoint:
        raise KeyError("Finetune checkpoint does not have a valid 'config' field")

    try:
        starting_checkpoint_fn = fine_tune_checkpoint["config"]["model"][
            "finetune_config"
        ]["starting_checkpoint"]
    except KeyError as e:
        logging.error(
            f"Finetune config missing entry config/model/finetune_config/starting_checkpoint {fine_tune_checkpoint['config']}"
        )
        raise e

    starting_checkpoint = torch.load(starting_checkpoint_fn, map_location="cpu")
    start_checkpoint_model_config = starting_checkpoint["config"]["model"]

    fine_tune_checkpoint["config"]["model"]["backbone"] = start_checkpoint_model_config[
        "backbone"
    ]
    # if we are data only, then copy over the heads config too
    ft_data_only = "heads" not in fine_tune_checkpoint["config"]["model"]
    if ft_data_only:
        fine_tune_checkpoint["config"]["model"]["heads"] = (
            start_checkpoint_model_config["heads"]
        )

    fine_tune_checkpoint["config"]["model"].pop("finetune_config")

    torch.save(fine_tune_checkpoint, output_checkpoint_fn)

    if fine_tune_yaml_fn is not None:
        with open(fine_tune_yaml_fn) as yaml_f:
            fine_tune_yaml = yaml.safe_load(yaml_f)
        fine_tune_yaml["model"].pop("finetune_config")
        fine_tune_yaml["model"]["backbone"] = start_checkpoint_model_config["backbone"]
        if ft_data_only:
            fine_tune_yaml["model"]["heads"] = start_checkpoint_model_config["heads"]
        with open(output_yaml_fn, "w") as yaml_file:
            yaml.dump(fine_tune_yaml, yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tune-checkpoint",
        help="path to fine tuned checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-release-checkpoint",
        help="path to output checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fine-tune-yaml",
        help="path to fine tune yaml config",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output-release-yaml",
        help="path to output yaml config",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    convert_fine_tune_checkpoint(
        fine_tune_yaml_fn=args.fine_tune_yaml,
        fine_tune_checkpoint_fn=args.fine_tune_checkpoint,
        output_checkpoint_fn=args.output_release_checkpoint,
        output_yaml_fn=args.output_release_yaml,
    )
