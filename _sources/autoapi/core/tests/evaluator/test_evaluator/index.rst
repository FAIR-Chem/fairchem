:py:mod:`core.tests.evaluator.test_evaluator`
=============================================

.. py:module:: core.tests.evaluator.test_evaluator

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.tests.evaluator.test_evaluator.TestMetrics
   core.tests.evaluator.test_evaluator.TestS2EFEval
   core.tests.evaluator.test_evaluator.TestIS2RSEval
   core.tests.evaluator.test_evaluator.TestIS2REEval



Functions
~~~~~~~~~

.. autoapisummary::

   core.tests.evaluator.test_evaluator.load_evaluator_s2ef
   core.tests.evaluator.test_evaluator.load_evaluator_is2rs
   core.tests.evaluator.test_evaluator.load_evaluator_is2re



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



