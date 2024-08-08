from __future__ import annotations

from abc import ABC
import copy
from enum import Enum
import errno
import os
from typing import TYPE_CHECKING
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.models.base import BackboneInterface, HeadInterface, HydraInterface
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
    assert (
        checkpoint["config"]["model"]["name"] == "hydra"
    ), "Can only finetune a hydra model"
    return checkpoint["config"]["model"]


def load_hydra_model(checkpoint_path: str) -> HydraInterface:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            errno.ENOENT, "Checkpoint file not found", checkpoint_path
        )
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    config_copy = copy.deepcopy(checkpoint["config"]["model"])
    name = config_copy.pop("name")
    hydra_model = registry.get_model_class(name)(**config_copy)
    assert isinstance(
        hydra_model, HydraInterface
    ), "Can only load models with the HydraInterface"
    matched_dict = match_state_dict(hydra_model.state_dict(), checkpoint["state_dict"])
    load_state_dict(hydra_model, matched_dict, strict=True)
    return hydra_model


class FTConfig:
    FT_CONFIG_NAME = "finetune_config"
    CHECKPOINT_PROPERTY = "starting_checkpoint"
    STARTING_MODEL = "starting_model"
    MODE = "mode"
    HEADS = "heads"

    def __init__(self, config: dict):
        self.config = config
        self._mode = FineTuneMode[self.config[FTConfig.MODE]]
        assert (
            (FTConfig.CHECKPOINT_PROPERTY in self.config)
            ^ (FTConfig.STARTING_MODEL in self.config)
        ), "Only one of checkpoint or starting model config can be provided to FineTuneHydra!"
        assert FTConfig.MODE in self.config
        if self._mode == FineTuneMode.RETAIN_BACKBONE_ONLY:
            # in this mode, we keep the backbone but attach new output heads specified in head config
            assert (
                FTConfig.HEADS in self.config
            ), "heads cannot be empty when using RETAIN_BACKBONE_ONLY mode!"

    def load_model(self) -> nn.Module:
        # if provided a checkpoint to start then load the model and weights from the given checkpoint
        if FTConfig.CHECKPOINT_PROPERTY in self.config:
            hydra_model: HydraInterface = load_hydra_model(
                self.config[FTConfig.CHECKPOINT_PROPERTY]
            )
        # if provided a hydra config to start, build from the starting hydra model
        elif FTConfig.STARTING_MODEL in self.config:
            # register model from hydra_config
            config_copy = copy.deepcopy(self.config[FTConfig.STARTING_MODEL])
            name = config_copy.pop("name")
            hydra_model = registry.get_model_class(name)(**config_copy)
            assert isinstance(hydra_model, HydraInterface)

        num_params = sum(p.numel() for p in hydra_model.parameters())
        logging.info(f"Loaded Original hydra model with {num_params} params")
        return hydra_model

    def get_standalone_config(self) -> dict:
        # replace a config with a checkpoint with one that has the model config only
        # this is required for standalone prediction (so we don't need to ship the original checkpoint),
        # multi-round finetuning, and better robustness
        standalone_config = {
            "name": "finetune_hydra",
            FTConfig.FT_CONFIG_NAME: self.config,
        }
        if FTConfig.CHECKPOINT_PROPERTY in self.config:
            # modify the config to store the original model config inside model attrs so we dont need the checkpoint again when loading from checkpoint
            new_config = copy.deepcopy(self.config)
            new_config[FTConfig.STARTING_MODEL] = (
                get_hydra_model_config_from_checkpoint(
                    self.config[FTConfig.CHECKPOINT_PROPERTY]
                )
            )
            del new_config[FTConfig.CHECKPOINT_PROPERTY]
            standalone_config[FTConfig.FT_CONFIG_NAME] = new_config
            return standalone_config
        else:
            return standalone_config

    @property
    def mode(self) -> FineTuneMode:
        return self._mode

    @property
    def head_config(self) -> dict:
        return copy.deepcopy(self.config[FTConfig.HEADS])


class FineTuneModelInterface(ABC):
    def __init__(self, finetune_config: dict):
        pass


@registry.register_model("finetune_hydra")
class FineTuneHydra(nn.Module, HydraInterface, FineTuneModelInterface):
    def __init__(self, finetune_config: dict):
        super().__init__()
        ft_config = FTConfig(finetune_config)
        logging.info(f"Initializing FineTuneHydra model in {ft_config.mode} mode")
        hydra_model: HydraInterface = ft_config.load_model()
        self.backbone: BackboneInterface = hydra_model.get_backbone()

        if ft_config.mode == FineTuneMode.DATA_ONLY:
            # in this mode, we just use the model as is and train on it with new data
            self.output_heads: dict[str, HeadInterface] = hydra_model.get_output_heads()
        elif ft_config.mode == FineTuneMode.RETAIN_BACKBONE_ONLY:
            # in this mode, we keep the backbone but attach new output heads specified in head config
            self.output_heads: dict[str, HeadInterface] = {}
            heads_config = ft_config.head_config
            head_names_sorted = sorted(heads_config.keys())
            for head_name in head_names_sorted:
                head_config = heads_config[head_name]
                if "module" not in head_config:
                    raise ValueError(
                        f"{head_name} head does not specify module to use for the head"
                    )

                module_name = head_config.pop("module")
                self.output_heads[head_name] = registry.get_model_class(module_name)(
                    self.backbone,
                    **head_config,
                )
                num_params = sum(
                    p.numel() for p in self.output_heads[head_name].parameters()
                )
                logging.info(
                    f"Attaching new output head: {module_name} with {num_params} params"
                )
            self.output_heads = torch.nn.ModuleDict(self.output_heads)

    def forward(self, data: Batch):
        emb = self.backbone(data)
        out = {}
        for k in self.output_heads.keys():
            out.update(self.output_heads[k](data, emb))
        return out

    def get_backbone(self) -> BackboneInterface:
        return self.backbone

    def get_heads(self) -> dict[str, HeadInterface]:
        return self.output_heads
