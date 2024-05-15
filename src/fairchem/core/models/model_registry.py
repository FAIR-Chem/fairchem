"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import shutil
from importlib import resources
from typing import TYPE_CHECKING

import urllib3
import yaml

from fairchem.core import models

if TYPE_CHECKING:
    from pathlib import Path


with (resources.files(models) / "pretrained_models.yml").open("rt") as f:
    MODEL_REGISTRY = yaml.safe_load(f)


available_pretrained_models = tuple(MODEL_REGISTRY.keys())


def model_name_to_local_file(model_name: str, local_cache: str | Path) -> str:
    """Download a pretrained checkpoint if it does not exist already

    Args:
        model_name (str): the model name. See available_pretrained_checkpoints.
        local_cache (str or Path): path to local cache directory

    Returns:
        str: local path to checkpoint file
    """
    logging.info(f"Checking local cache: {local_cache} for model {model_name}")
    if model_name not in MODEL_REGISTRY:
        logging.error(f"Not a valid model name '{model_name}'")
        raise ValueError(
            f"Not a valid model name '{model_name}'. Model name must be one of {available_pretrained_models}"
        )
    if not os.path.exists(local_cache):
        os.makedirs(local_cache, exist_ok=True)
    if not os.path.exists(local_cache):
        logging.error(f"Failed to create local cache folder '{local_cache}'")
        raise RuntimeError(f"Failed to create local cache folder '{local_cache}'")
    model_url = MODEL_REGISTRY[model_name]
    local_path = os.path.join(local_cache, os.path.basename(model_url))

    # download the file
    if not os.path.isfile(local_path):
        local_path_tmp = local_path + ".tmp"  # download to a tmp file in case we fail
        http = urllib3.PoolManager()
        with open(local_path_tmp, "wb") as out:
            r = http.request("GET", model_url, preload_content=False)
            shutil.copyfileobj(r, out)
        shutil.move(local_path_tmp, local_path)
    return local_path
