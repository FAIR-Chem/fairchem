:py:mod:`core.tests.preprocessing.test_radius_graph_pbc`
========================================================

.. py:module:: core.tests.preprocessing.test_radius_graph_pbc

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.tests.preprocessing.test_radius_graph_pbc.TestRadiusGraphPBC



Functions
~~~~~~~~~

.. autoapisummary::

   core.tests.preprocessing.test_radius_graph_pbc.load_data
   core.tests.preprocessing.test_radius_graph_pbc.check_features_match



.. py:function:: load_data(request) -> None


.. py:function:: check_features_match(edge_index_1, cell_offsets_1, edge_index_2, cell_offsets_2) -> bool


.. py:class:: TestRadiusGraphPBC


   .. py:method:: test_radius_graph_pbc() -> None


   .. py:method:: test_bulk() -> None


   .. py:method:: test_molecule() -> None



