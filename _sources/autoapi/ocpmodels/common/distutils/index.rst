:py:mod:`ocpmodels.common.distutils`
====================================

.. py:module:: ocpmodels.common.distutils

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.distutils.os_environ_get_or_throw
   ocpmodels.common.distutils.setup
   ocpmodels.common.distutils.cleanup
   ocpmodels.common.distutils.initialized
   ocpmodels.common.distutils.get_rank
   ocpmodels.common.distutils.get_world_size
   ocpmodels.common.distutils.is_master
   ocpmodels.common.distutils.synchronize
   ocpmodels.common.distutils.broadcast
   ocpmodels.common.distutils.all_reduce
   ocpmodels.common.distutils.all_gather



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


.. py:function:: all_gather(data, group=dist.group.WORLD, device=None) -> List[torch.Tensor]


