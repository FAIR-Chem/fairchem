core.common.data_parallel
=========================

.. py:module:: core.common.data_parallel

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.common.data_parallel.OCPCollater
   core.common.data_parallel.StatefulDistributedSampler
   core.common.data_parallel.BalancedBatchSampler


Functions
---------

.. autoapisummary::

   core.common.data_parallel._balanced_partition
   core.common.data_parallel._ensure_supported


Module Contents
---------------

.. py:class:: OCPCollater(otf_graph: bool = False)

   .. py:attribute:: otf_graph


   .. py:method:: __call__(data_list: list[torch_geometric.data.Data]) -> torch_geometric.data.Batch


.. py:function:: _balanced_partition(sizes: numpy.typing.NDArray[numpy.int_], num_parts: int)

   Greedily partition the given set by always inserting
   the largest element into the smallest partition.


.. py:class:: StatefulDistributedSampler(dataset, batch_size, **kwargs)

   Bases: :py:obj:`torch.utils.data.DistributedSampler`


   More fine-grained state DataSampler that uses training iteration and epoch
   both for shuffling data. PyTorch DistributedSampler only uses epoch
   for the shuffling and starts sampling data from the start. In case of training
   on very large data, we train for one epoch only and when we resume training,
   we want to resume the data sampler from the training iteration.


   .. py:attribute:: start_iter
      :value: 0



   .. py:attribute:: batch_size


   .. py:method:: __iter__()


   .. py:method:: set_epoch_and_start_iteration(epoch, start_iter)


.. py:function:: _ensure_supported(dataset: Any)

.. py:class:: BalancedBatchSampler(dataset: torch.utils.data.Dataset, *, batch_size: int, num_replicas: int, rank: int, device: torch.device, seed: int, mode: bool | Literal['atoms'] = 'atoms', shuffle: bool = True, on_error: Literal['warn_and_balance', 'warn_and_no_balance', 'raise'] = 'raise', drop_last: bool = False)

   Bases: :py:obj:`torch.utils.data.BatchSampler`


   Wraps another sampler to yield a mini-batch of indices.

   :param sampler: Base sampler. Can be any iterable object
   :type sampler: Sampler or Iterable
   :param batch_size: Size of mini-batch.
   :type batch_size: int
   :param drop_last: If ``True``, the sampler will drop the last batch if
                     its size would be less than ``batch_size``
   :type drop_last: bool

   .. rubric:: Example

   >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
   [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
   >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
   [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


   .. py:attribute:: disabled
      :value: False



   .. py:attribute:: on_error


   .. py:attribute:: device


   .. py:method:: _get_natoms(batch_idx: list[int])


   .. py:method:: set_epoch_and_start_iteration(epoch: int, start_iteration: int) -> None


   .. py:method:: set_epoch(epoch: int) -> None


   .. py:method:: _dist_enabled()
      :staticmethod:



   .. py:method:: __iter__()


