:py:mod:`ocpmodels.common.transforms`
=====================================

.. py:module:: ocpmodels.common.transforms

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.common.transforms.RandomRotate




.. py:class:: RandomRotate(degrees, axes: List[int] = [0, 1, 2])


   Rotates node positions around a specific axis by a randomly sampled
   factor within a given interval.

   :param degrees: Rotation interval from which the rotation
                   angle is sampled. If `degrees` is a number instead of a
                   tuple, the interval is given by :math:`[-\mathrm{degrees},
                   \mathrm{degrees}]`.
   :type degrees: tuple or float
   :param axes: The rotation axes. (default: `[0, 1, 2]`)
   :type axes: int, optional

   .. py:method:: __call__(data)


   .. py:method:: __repr__() -> str

      Return repr(self).



