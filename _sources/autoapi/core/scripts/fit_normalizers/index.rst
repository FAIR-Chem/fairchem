core.scripts.fit_normalizers
============================

.. py:module:: core.scripts.fit_normalizers

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.scripts.fit_normalizers.parser


Functions
---------

.. autoapisummary::

   core.scripts.fit_normalizers.fit_norms


Module Contents
---------------

.. py:function:: fit_norms(config: dict, output_path: str | pathlib.Path, linref_file: str | pathlib.Path | None = None, linref_target: str = 'energy') -> None

   Fit dataset mean and std using the standard config

   :param config: config
   :param output_path: output path
   :param linref_file: path to fitted linear references. IF these are used in training they must be used to compute mean/std
   :param linref_target: target using linear references, basically always energy.


.. py:data:: parser

