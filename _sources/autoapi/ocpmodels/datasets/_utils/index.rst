:py:mod:`ocpmodels.datasets._utils`
===================================

.. py:module:: ocpmodels.datasets._utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.datasets._utils.rename_data_object_keys



.. py:function:: rename_data_object_keys(data_object: torch_geometric.data.Data, key_mapping: dict[str, str]) -> torch_geometric.data.Data

   Rename data object keys

   :param data_object: data object
   :param key_mapping: dictionary specifying keys to rename and new names {prev_key: new_key}


