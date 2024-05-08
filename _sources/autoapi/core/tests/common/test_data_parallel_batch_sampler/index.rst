:py:mod:`core.tests.common.test_data_parallel_batch_sampler`
============================================================

.. py:module:: core.tests.common.test_data_parallel_batch_sampler


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.tests.common.test_data_parallel_batch_sampler._temp_file
   core.tests.common.test_data_parallel_batch_sampler.valid_path_dataset
   core.tests.common.test_data_parallel_batch_sampler.invalid_path_dataset
   core.tests.common.test_data_parallel_batch_sampler.invalid_dataset
   core.tests.common.test_data_parallel_batch_sampler.test_lowercase
   core.tests.common.test_data_parallel_batch_sampler.test_invalid_mode
   core.tests.common.test_data_parallel_batch_sampler.test_invalid_dataset
   core.tests.common.test_data_parallel_batch_sampler.test_invalid_path_dataset
   core.tests.common.test_data_parallel_batch_sampler.test_valid_dataset
   core.tests.common.test_data_parallel_batch_sampler.test_disabled
   core.tests.common.test_data_parallel_batch_sampler.test_single_node
   core.tests.common.test_data_parallel_batch_sampler.test_stateful_distributed_sampler_noshuffle
   core.tests.common.test_data_parallel_batch_sampler.test_stateful_distributed_sampler_vs_distributed_sampler
   core.tests.common.test_data_parallel_batch_sampler.test_stateful_distributed_sampler
   core.tests.common.test_data_parallel_batch_sampler.test_stateful_distributed_sampler_numreplicas
   core.tests.common.test_data_parallel_batch_sampler.test_stateful_distributed_sampler_numreplicas_drop_last



Attributes
~~~~~~~~~~

.. autoapisummary::

   core.tests.common.test_data_parallel_batch_sampler.DATA
   core.tests.common.test_data_parallel_batch_sampler.SIZE_ATOMS
   core.tests.common.test_data_parallel_batch_sampler.SIZE_NEIGHBORS
   core.tests.common.test_data_parallel_batch_sampler.T_co


.. py:data:: DATA
   :value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

   

.. py:data:: SIZE_ATOMS
   :value: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

   

.. py:data:: SIZE_NEIGHBORS
   :value: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

   

.. py:data:: T_co

   

.. py:function:: _temp_file(name: str)


.. py:function:: valid_path_dataset()


.. py:function:: invalid_path_dataset()


.. py:function:: invalid_dataset()


.. py:function:: test_lowercase(invalid_dataset) -> None


.. py:function:: test_invalid_mode(invalid_dataset) -> None


.. py:function:: test_invalid_dataset(invalid_dataset) -> None


.. py:function:: test_invalid_path_dataset(invalid_path_dataset) -> None


.. py:function:: test_valid_dataset(valid_path_dataset) -> None


.. py:function:: test_disabled(valid_path_dataset) -> None


.. py:function:: test_single_node(valid_path_dataset) -> None


.. py:function:: test_stateful_distributed_sampler_noshuffle(valid_path_dataset) -> None


.. py:function:: test_stateful_distributed_sampler_vs_distributed_sampler(valid_path_dataset) -> None


.. py:function:: test_stateful_distributed_sampler(valid_path_dataset) -> None


.. py:function:: test_stateful_distributed_sampler_numreplicas(valid_path_dataset) -> None


.. py:function:: test_stateful_distributed_sampler_numreplicas_drop_last(valid_path_dataset) -> None


