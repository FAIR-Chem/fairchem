core.components.utils
=====================

.. py:module:: core.components.utils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.utils.ManagedAttribute


Module Contents
---------------

.. py:class:: ManagedAttribute(enforced_type: type | None = None)

   A descriptor helper to manage setting/access to an attribute of a class


   .. py:attribute:: _enforced_type


   .. py:method:: __set_name__(owner: fairchem.core.components.runner.Runner | fairchem.core.components.reducer.Reducer, name: str)


   .. py:method:: __get__(obj: fairchem.core.components.runner.Runner | fairchem.core.components.reducer.Reducer, objtype: type[fairchem.core.components.runner.Runner | fairchem.core.components.reducer.Reducer] | None = None)


   .. py:method:: __set__(obj: fairchem.core.components.runner.Runner | fairchem.core.components.reducer.Reducer, value: Any)


