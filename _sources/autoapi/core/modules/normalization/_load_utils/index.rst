core.modules.normalization._load_utils
======================================

.. py:module:: core.modules.normalization._load_utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.modules.normalization._load_utils._load_check_duplicates
   core.modules.normalization._load_utils._load_from_config


Module Contents
---------------

.. py:function:: _load_check_duplicates(config: dict, name: str) -> dict[str, torch.nn.Module]

   Attempt to load a single file with normalizers/element references and check config for duplicate targets.

   :param config: configuration dictionary
   :param name: Name of module to use for logging

   :returns: dictionary of normalizer or element reference modules


.. py:function:: _load_from_config(config: dict, name: str, fit_fun: Callable[[list[str], torch.utils.data.Dataset, Any, Ellipsis], dict[str, torch.nn.Module]], create_fun: Callable[[str | pathlib.Path], torch.nn.Module], dataset: torch.utils.data.Dataset, checkpoint_dir: str | pathlib.Path | None = None, **fit_kwargs) -> dict[str, torch.nn.Module]

   Load or fit normalizers or element references from config

   If a fit is done, a fitted key with value true is added to the config to avoid re-fitting
   once a checkpoint has been saved.

   :param config: configuration dictionary
   :param name: Name of module to use for logging
   :param fit_fun: Function to fit modules
   :param create_fun: Function to create a module from file
   :param checkpoint_dir: directory to save modules. If not given, modules won't be saved.

   :returns: dictionary of normalizer or element reference modules


