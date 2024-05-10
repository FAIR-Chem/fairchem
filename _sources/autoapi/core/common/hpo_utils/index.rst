:py:mod:`core.common.hpo_utils`
===============================

.. py:module:: core.common.hpo_utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.common.hpo_utils.tune_reporter
   core.common.hpo_utils.label_metric_dict



.. py:function:: tune_reporter(iters, train_metrics, val_metrics, test_metrics=None, metric_to_opt: str = 'val_loss', min_max: str = 'min') -> None

   Wrapper function for tune.report()

   :param iters: dict with training iteration info (e.g. steps, epochs)
   :type iters: dict
   :param train_metrics: train metrics dict
   :type train_metrics: dict
   :param val_metrics: val metrics dict
   :type val_metrics: dict
   :param test_metrics: test metrics dict, default is None
   :type test_metrics: dict, optional
   :param metric_to_opt: str for val metric to optimize, default is val_loss
   :type metric_to_opt: str, optional
   :param min_max: either "min" or "max", determines whether metric_to_opt is to be minimized or maximized, default is min
   :type min_max: str, optional


.. py:function:: label_metric_dict(metric_dict, split)


