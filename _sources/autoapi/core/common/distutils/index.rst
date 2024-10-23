core.common.distutils
=====================

.. py:module:: core.common.distutils

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.common.distutils.T
   core.common.distutils.DISTRIBUTED_PORT
   core.common.distutils.CURRENT_DEVICE_STR


Functions
---------

.. autoapisummary::

   core.common.distutils.os_environ_get_or_throw
   core.common.distutils.setup
   core.common.distutils.cleanup
   core.common.distutils.initialized
   core.common.distutils.get_rank
   core.common.distutils.get_world_size
   core.common.distutils.is_master
   core.common.distutils.synchronize
   core.common.distutils.broadcast
   core.common.distutils.broadcast_object_list
   core.common.distutils.all_reduce
   core.common.distutils.all_gather
   core.common.distutils.gather_objects
   core.common.distutils.assign_device_for_local_rank
   core.common.distutils.get_device_for_local_rank
   core.common.distutils.setup_env_local


Module Contents
---------------

.. py:data:: T

.. py:data:: DISTRIBUTED_PORT
   :value: 13356


.. py:data:: CURRENT_DEVICE_STR
   :value: 'CURRRENT_DEVICE'


.. py:function:: os_environ_get_or_throw(x: str) -> str

.. py:function:: setup(config) -> None

.. py:function:: cleanup() -> None

.. py:function:: initialized() -> bool

.. py:function:: get_rank() -> int

.. py:function:: get_world_size() -> int

.. py:function:: is_master() -> bool

.. py:function:: synchronize() -> None

.. py:function:: broadcast(tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False) -> None

.. py:function:: broadcast_object_list(object_list: list[Any], src: int, group=dist.group.WORLD, device: str | None = None) -> None

.. py:function:: all_reduce(data, group=dist.group.WORLD, average: bool = False, device=None) -> torch.Tensor

.. py:function:: all_gather(data, group=dist.group.WORLD, device=None) -> list[torch.Tensor]

.. py:function:: gather_objects(data: T, group: torch.distributed.ProcessGroup = dist.group.WORLD) -> list[T]

   Gather a list of pickleable objects into rank 0


.. py:function:: assign_device_for_local_rank(cpu: bool, local_rank: int)

.. py:function:: get_device_for_local_rank()

.. py:function:: setup_env_local()

