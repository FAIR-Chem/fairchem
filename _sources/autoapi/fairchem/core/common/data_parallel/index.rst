:py:mod:`fairchem.core.common.data_parallel`
============================================

.. py:module:: fairchem.core.common.data_parallel

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.common.data_parallel.OCPCollater
   fairchem.core.common.data_parallel._HasMetadata
   fairchem.core.common.data_parallel.StatefulDistributedSampler
   fairchem.core.common.data_parallel.BalancedBatchSampler



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.common.data_parallel.balanced_partition



.. py:class:: OCPCollater(otf_graph: bool = False)


   .. py:method:: __call__(data_list: list[torch_geometric.data.Data]) -> torch_geometric.data.Batch



.. py:function:: balanced_partition(sizes: numpy.typing.NDArray[numpy.int_], num_parts: int)

   Greedily partition the given set by always inserting
   the largest element into the smallest partition.


.. py:class:: _HasMetadata


   Bases: :py:obj:`Protocol`

   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto(Protocol[T]):
           def meth(self) -> T:
               ...

   .. py:property:: metadata_path
      :type: pathlib.Path



.. py:class:: StatefulDistributedSampler(dataset, batch_size, **kwargs)


   Bases: :py:obj:`torch.utils.data.DistributedSampler`

   More fine-grained state DataSampler that uses training iteration and epoch
   both for shuffling data. PyTorch DistributedSampler only uses epoch
   for the shuffling and starts sampling data from the start. In case of training
   on very large data, we train for one epoch only and when we resume training,
   we want to resume the data sampler from the training iteration.

   .. py:method:: __iter__()


   .. py:method:: set_epoch_and_start_iteration(epoch, start_iter)



.. py:class:: BalancedBatchSampler(dataset, batch_size: int, num_replicas: int, rank: int, device: torch.device, mode: str | bool = 'atoms', shuffle: bool = True, drop_last: bool = False, force_balancing: bool = False, throw_on_error: bool = False)


   Bases: :py:obj:`torch.utils.data.Sampler`

   Base class for all Samplers.

   Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
   way to iterate over indices or lists of indices (batches) of dataset elements, and a :meth:`__len__` method
   that returns the length of the returned iterators.

   :param data_source: This argument is not used and will be removed in 2.2.0.
                       You may still have custom implementation that utilizes it.
   :type data_source: Dataset

   .. rubric:: Example

   >>> # xdoctest: +SKIP
   >>> class AccedingSequenceLengthSampler(Sampler[int]):
   >>>     def __init__(self, data: List[str]) -> None:
   >>>         self.data = data
   >>>
   >>>     def __len__(self) -> int:
   >>>         return len(self.data)
   >>>
   >>>     def __iter__(self) -> Iterator[int]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         yield from torch.argsort(sizes).tolist()
   >>>
   >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
   >>>     def __init__(self, data: List[str], batch_size: int) -> None:
   >>>         self.data = data
   >>>         self.batch_size = batch_size
   >>>
   >>>     def __len__(self) -> int:
   >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
   >>>
   >>>     def __iter__(self) -> Iterator[List[int]]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
   >>>             yield batch.tolist()

   .. note:: The :meth:`__len__` method isn't strictly required by
             :class:`~torch.utils.data.DataLoader`, but is expected in any
             calculation involving the length of a :class:`~torch.utils.data.DataLoader`.

   .. py:method:: _load_dataset(dataset, mode: Literal[atoms, neighbors])


   .. py:method:: __len__() -> int


   .. py:method:: set_epoch_and_start_iteration(epoch: int, start_iteration: int) -> None


   .. py:method:: __iter__()



