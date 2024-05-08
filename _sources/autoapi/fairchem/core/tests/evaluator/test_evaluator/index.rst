:py:mod:`fairchem.core.tests.evaluator.test_evaluator`
======================================================

.. py:module:: fairchem.core.tests.evaluator.test_evaluator

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.tests.evaluator.test_evaluator.TestMetrics
   fairchem.core.tests.evaluator.test_evaluator.TestS2EFEval
   fairchem.core.tests.evaluator.test_evaluator.TestIS2RSEval
   fairchem.core.tests.evaluator.test_evaluator.TestIS2REEval



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.tests.evaluator.test_evaluator.load_evaluator_s2ef
   fairchem.core.tests.evaluator.test_evaluator.load_evaluator_is2rs
   fairchem.core.tests.evaluator.test_evaluator.load_evaluator_is2re



.. py:function:: load_evaluator_s2ef(request) -> None


.. py:function:: load_evaluator_is2rs(request) -> None


.. py:function:: load_evaluator_is2re(request) -> None


.. py:class:: TestMetrics


   .. py:method:: test_cosine_similarity() -> None


   .. py:method:: test_magnitude_error() -> None



.. py:class:: TestS2EFEval


   .. py:method:: test_metrics_exist() -> None



.. py:class:: TestIS2RSEval


   .. py:method:: test_metrics_exist() -> None



.. py:class:: TestIS2REEval


   .. py:method:: test_metrics_exist() -> None



