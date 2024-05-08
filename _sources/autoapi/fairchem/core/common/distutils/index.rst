:py:mod:`fairchem.core.common.distutils`
========================================

.. py:module:: fairchem.core.common.distutils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.common.distutils.os_environ_get_or_throw
   fairchem.core.common.distutils.setup
   fairchem.core.common.distutils.cleanup
   fairchem.core.common.distutils.initialized
   fairchem.core.common.distutils.get_rank
   fairchem.core.common.distutils.get_world_size
   fairchem.core.common.distutils.is_master
   fairchem.core.common.distutils.synchronize
   fairchem.core.common.distutils.broadcast
   fairchem.core.common.distutils.all_reduce
   fairchem.core.common.distutils.all_gather
   fairchem.core.common.distutils.gather_objects



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.common.distutils.T


.. py:data:: T

   

.. py:function:: os_environ_get_or_throw(x: str) -> str


.. py:function:: setup(config) -> None


.. py:function:: cleanup() -> None


.. py:function:: initialized() -> bool


.. py:function:: get_rank() -> int


.. py:function:: get_world_size() -> int


.. py:function:: is_master() -> bool


.. py:function:: synchronize() -> None


.. py:function:: broadcast(tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False) -> None


.. py:function:: all_reduce(data, group=dist.group.WORLD, average: bool = False, device=None) -> torch.Tensor


.. py:function:: all_gather(data, group=dist.group.WORLD, device=None) -> list[torch.Tensor]


.. py:function:: gather_objects(data: T, group: torch.distributed.ProcessGroup = dist.group.WORLD) -> list[T]

   Gather a list of pickleable objects into rank 0


