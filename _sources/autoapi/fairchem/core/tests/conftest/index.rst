:py:mod:`fairchem.core.tests.conftest`
======================================

.. py:module:: fairchem.core.tests.conftest

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.tests.conftest.Approx
   fairchem.core.tests.conftest._ApproxNumpyFormatter
   fairchem.core.tests.conftest.ApproxExtension



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.tests.conftest._try_parse_approx
   fairchem.core.tests.conftest.snapshot



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.tests.conftest.DEFAULT_RTOL
   fairchem.core.tests.conftest.DEFAULT_ATOL


.. py:data:: DEFAULT_RTOL
   :value: 0.001

   

.. py:data:: DEFAULT_ATOL
   :value: 0.001

   

.. py:class:: Approx(data: numpy.ndarray | list, *, rtol: float | None = None, atol: float | None = None)


   Wrapper object for approximately compared numpy arrays.

   .. py:method:: __repr__() -> str

      Return repr(self).



.. py:class:: _ApproxNumpyFormatter(data)


   .. py:method:: __repr__() -> str

      Return repr(self).



.. py:function:: _try_parse_approx(data: syrupy.types.SerializableData) -> Approx | None

   Parse the string representation of an Approx object.
   We can just use eval here, since we know the string is safe.


.. py:class:: ApproxExtension


   Bases: :py:obj:`syrupy.extensions.amber.AmberSnapshotExtension`

   By default, syrupy uses the __repr__ of the expected (snapshot) and actual values
   to serialize them into strings. Then, it compares the strings to see if they match.

   However, this behavior is not ideal for comparing floats/ndarrays. For example,
   if we have a snapshot with a float value of 0.1, and the actual value is 0.10000000000000001,
   then the strings will not match, even though the values are effectively equal.

   To work around this, we override the serialize method to seralize the expected value
   into a special representation. Then, we override the matches function (which originally does a
   simple string comparison) to parse the expected and actual values into numpy arrays.
   Finally, we compare the arrays using np.allclose.

   .. py:method:: matches(*, serialized_data: syrupy.types.SerializableData, snapshot_data: syrupy.types.SerializableData) -> bool


   .. py:method:: serialize(data, **kwargs)



.. py:function:: snapshot(snapshot)


