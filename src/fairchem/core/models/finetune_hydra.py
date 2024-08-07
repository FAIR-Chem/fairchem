from abc import ABC
import copy
from enum import Enum
import errno
import os
from typing import TYPE_CHECKING
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from torch import nn
import torch
import logging

if TYPE_CHECKING:
    from torch_geometric.data import Batch

class FineTuneMode(Enum):
    # in DATA_ONLY, we load the entire model and only finetune on new data
    DATA_ONLY = 1
    # in this mode, we only load the Backbone and feed the output of the backbone
    # to new heads that are specified
    RETAIN_BACKBONE_ONLY = 2
    
def get_hydra_model_config_from_checkpoint(checkpoint_path: str) -> dict:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            errno.ENOENT, "Checkpoint file not found", checkpoint_path
        )
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["config"]["model"]["name"] == "hydra", "Can only finetune a hydra model"
    return checkpoint["config"]["model"]


def load_hydra_model(checkpoint_path: str) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            errno.ENOENT, "Checkpoint file not found", checkpoint_path
        )
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["config"]["model"]["name"] == "hydra", "Can only finetune a hydra model"
    config_copy = copy.deepcopy(checkpoint["config"]["model"])
    config_copy.pop("name")
    hydra_model = registry.get_model_class("hydra")(**config_copy)
    matched_dict = match_state_dict(hydra_model.state_dict(), checkpoint["state_dict"])
    load_state_dict(hydra_model, matched_dict, strict=True)
    return hydra_model


class FTStartingConfig:
    CHECKPOINT_PROPERTY = "starting_checkpoint"
    STARTING_MODEL = "starting_model"
    STARTING_CONFIG = "starting_config"

    def __init__(self, config: dict):
        self.config = config
        assert (FTStartingConfig.CHECKPOINT_PROPERTY in self.config) ^ (FTStartingConfig.STARTING_MODEL in self.config), \
            "Only one of checkpoint or starting model config can be provided to FineTuneHydra!"

    def load_model(self) -> nn.Module:
        # if provided a checkpoint to start then load the model and weights from the given checkpoint
        if FTStartingConfig.CHECKPOINT_PROPERTY in self.config:
            hydra_model = load_hydra_model(self.config[FTStartingConfig.CHECKPOINT_PROPERTY])
        # if provided a hydra config to start, build from the starting hydra model
        elif FTStartingConfig.STARTING_MODEL in self.config:
            # register model from hydra_config
            config_copy = copy.deepcopy(self.config[FTStartingConfig.STARTING_MODEL])
            name = config_copy.pop("name")
            hydra_model = registry.get_model_class(name)(**config_copy)

        num_params = sum(p.numel() for p in hydra_model.parameters())
        logging.info(f"Loaded Original hydra model with {num_params} params")
        return hydra_model

    def get_standalone_config(self) -> dict:
        # replace a config with a checkpoint with one that has the model config only
        if FTStartingConfig.CHECKPOINT_PROPERTY in self.config:
            # modify the config to store the original model config inside model attrs so we dont need the checkpoint again when loading from checkpoint
            new_config = copy.deepcopy(self.config)
            new_config[FTStartingConfig.STARTING_CONFIG] = get_hydra_model_config_from_checkpoint(self.config[FTStartingConfig.CHECKPOINT_PROPERTY])
            del new_config[FTStartingConfig.CHECKPOINT_PROPERTY]
            return new_config
        else:
            return self.config


class FineTuneModelInterface(ABC):
    def __init__(self, 
                 mode: FineTuneMode,
                 finetune_config: dict,
                 tasks_config: dict | None = None):
        pass
        

@registry.register_model("finetune_hydra")
class FineTuneHydra(nn.Module, FineTuneModelInterface):
    def __init__(
        self,
        mode: FineTuneMode,
        starting_config: dict,
        tasks_config: dict | None = None,
    ):
        super().__init__()
        self.mode = FineTuneMode[mode]
        logging.info(f"Initializing FineTuneHydra model in {self.mode} mode")
        ft_config = FTStartingConfig(starting_config)
        hydra_model = ft_config.load_model()
        self.backbone = hydra_model.backbone

        if self.mode == FineTuneMode.DATA_ONLY:
            # in this mode, we just use the model as is and train on it with new data
            self.output_heads = hydra_model.output_heads
        elif self.mode == FineTuneMode.RETAIN_BACKBONE_ONLY:
            # in this mode, we keep the backbone but attach new output heads specified in tasks_config
            assert tasks_config, "tasks_config cannot be empty when using RETAIN_BACKBONE_ONLY mode!"
            # initialize new output heads, throw away old ones


    def forward(self, data):
        emb = self.backbone(data)
        out = {}
        for k in self.output_heads.keys():
            out.update(self.output_heads[k](data, emb))
        return out
