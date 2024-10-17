from __future__ import annotations

import logging
import os
from collections import OrderedDict
from copy import deepcopy

import torch
import yaml


def convert_checkpoint_and_config_to_hydra(
    yaml_fn, checkpoint_fn, new_yaml_fn, new_checkpoint_fn
):
    assert not os.path.exists(new_yaml_fn), "Output yaml cannot already exist!"
    assert not os.path.exists(
        new_checkpoint_fn
    ), "Output checkpoint cannot already exist!"

    def remove_module_prefix(x):
        while x[: len("module.")] == "module.":
            x = x[len("module.") :]
        return x

    def eqv2_state_dict_to_hydra_state_dict(eqv2_state_dict):
        hydra_state_dict = OrderedDict()
        for og_key in list(eqv2_state_dict.keys()):
            key_without_module = remove_module_prefix(og_key)
            if "force_block" in og_key or "energy_block" in og_key:
                key = "module." + key_without_module.replace(
                    "force_block", "output_heads.forces.force_block"
                ).replace("energy_block", "output_heads.energy.energy_block")
            else:
                key = "module.backbone." + key_without_module
            hydra_state_dict[key] = eqv2_state_dict[og_key]
        return hydra_state_dict

    def convert_configs_to_hydra(yaml_config, checkpoint_config):
        if isinstance(checkpoint_config["model"], str):
            name = checkpoint_config["model"]
            checkpoint_config["model"] = checkpoint_config.pop("model_attributes")
            checkpoint_config["model"]["name"] = name

        new_model_config = {
            "name": "hydra",
            "backbone": checkpoint_config["model"].copy(),
            "heads": {
                "energy": {"module": "equiformer_v2_energy_head"},
                "forces": {"module": "equiformer_v2_force_head"},
            },
        }
        assert new_model_config["backbone"]["name"] in ["equiformer_v2"]
        new_model_config["backbone"].pop("name")
        new_model_config["backbone"]["model"] = "equiformer_v2_backbone"

        # create a new checkpoint config
        new_checkpoint_config = deepcopy(checkpoint_config)
        new_checkpoint_config["model"] = new_model_config

        # create a new YAML config
        new_yaml_config = deepcopy(yaml_config)
        new_yaml_config["model"] = new_model_config

        for output_key, output_d in new_yaml_config["outputs"].items():
            if output_d["level"] == "system":
                output_d["property"] = "energy"
            elif output_d["level"] == "atom":
                output_d["property"] = "forces"
            else:
                logging.warning(
                    f"Converting output:{output_key} to new equiv2 hydra config \
                        failed to find level and could not set property in output correctly"
                )

        return new_yaml_config, new_checkpoint_config

    # load existing from disk
    with open(yaml_fn) as yaml_f:
        yaml_config = yaml.safe_load(yaml_f)
    checkpoint = torch.load(checkpoint_fn, map_location="cpu")

    new_checkpoint = checkpoint.copy()
    new_yaml_config, new_checkpoint_config = convert_configs_to_hydra(
        yaml_config, checkpoint["config"]
    )
    new_checkpoint["config"] = new_checkpoint_config
    new_checkpoint["state_dict"] = eqv2_state_dict_to_hydra_state_dict(
        checkpoint["state_dict"]
    )
    for key in ["ema", "optimizer", "scheduler"]:
        new_checkpoint.pop(key, None)

    # write output
    torch.save(new_checkpoint, new_checkpoint_fn)
    with open(str(new_yaml_fn), "w") as yaml_file:
        yaml.dump(new_yaml_config, yaml_file)
