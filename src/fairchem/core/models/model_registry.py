"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from importlib import resources
from typing import TYPE_CHECKING, Literal

import requests
from huggingface_hub import hf_hub_download
from pydantic import AnyUrl, BaseModel

from fairchem.core import models

if TYPE_CHECKING:
    from pathlib import Path


class HuggingFaceModel(BaseModel):
    type: Literal["huggingface_hub"]
    repo_id: Literal["fairchem/OMAT24"]
    filename: str


class URLModel(BaseModel):
    url: str
    type: Literal["url"]


class ModelRegistry(BaseModel):
    models: dict[str, AnyUrl | HuggingFaceModel | URLModel]


with (resources.files(models) / "pretrained_models.json").open("rb") as f:
    MODEL_REGISTRY = ModelRegistry(models=json.load(f))


available_pretrained_models = tuple(MODEL_REGISTRY.models.keys())


def model_name_to_local_file(model_name: str, local_cache: str | Path) -> str:
    """Download a pretrained checkpoint if it does not exist already

    Args:
        model_name (str): the model name. See available_pretrained_checkpoints.
        local_cache (str or Path): path to local cache directory

    Returns:
        str: local path to checkpoint file
    """
    logging.info(f"Checking local cache: {local_cache} for model {model_name}")
    if model_name not in available_pretrained_models:
        logging.error(f"Not a valid model name '{model_name}'")
        raise ValueError(
            f"Not a valid model name '{model_name}'. Model name must be one of {available_pretrained_models}"
        )

    if isinstance(MODEL_REGISTRY.models[model_name], URLModel):
        # We have a url to download

        if not os.path.exists(local_cache):
            os.makedirs(local_cache, exist_ok=True)
        if not os.path.exists(local_cache):
            logging.error(f"Failed to create local cache folder '{local_cache}'")
            raise RuntimeError(f"Failed to create local cache folder '{local_cache}'")
        model_url = MODEL_REGISTRY.models[model_name].url
        local_path = os.path.join(local_cache, os.path.basename(model_url))

        # download the file
        if not os.path.isfile(local_path):
            local_path_tmp = (
                local_path + ".tmp"
            )  # download to a tmp file in case we fail
            with open(local_path_tmp, "wb") as out:
                response = requests.get(model_url, stream=True)
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, out)
            shutil.move(local_path_tmp, local_path)
        return local_path
    elif isinstance(MODEL_REGISTRY.models[model_name], HuggingFaceModel):
        return hf_hub_download(
            repo_id=MODEL_REGISTRY.models[model_name].repo_id,
            filename=MODEL_REGISTRY.models[model_name].filename,
        )
    else:
        raise NotImplementedError(
            f"{type(MODEL_REGISTRY.models[model_name])} is an unknown registry type."
        )
