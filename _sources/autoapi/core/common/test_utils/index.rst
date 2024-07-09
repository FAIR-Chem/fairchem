core.common.test_utils
======================

.. py:module:: core.common.test_utils


Classes
-------

.. autoapisummary::

   core.common.test_utils.ForkedPdb
   core.common.test_utils.PGConfig


Functions
---------

.. autoapisummary::

   core.common.test_utils.spawn_multi_process
   core.common.test_utils._init_pg_and_rank_and_launch_test


Module Contents
---------------

.. py:class:: ForkedPdb(completekey='tab', stdin=None, stdout=None, skip=None, nosigint=False, readrc=True)

   Bases: :py:obj:`pdb.Pdb`


   A Pdb subclass that may be used from a forked multiprocessing child
   https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess/23654936#23654936

   example usage to debug a torch distributed run on rank 0:
   if torch.distributed.get_rank() == 0:
       from fairchem.core.common.test_utils import ForkedPdb
       ForkedPdb().set_trace()


   .. py:method:: interaction(*args, **kwargs)


.. py:class:: PGConfig

   .. py:attribute:: backend
      :type:  str


   .. py:attribute:: world_size
      :type:  int


   .. py:attribute:: gp_group_size
      :type:  int
      :value: 1



   .. py:attribute:: port
      :type:  str
      :value: '12345'



   .. py:attribute:: use_gp
      :type:  bool
      :value: True



.. py:function:: spawn_multi_process(config: PGConfig, test_method: callable, *test_method_args: Any, **test_method_kwargs: Any) -> list[Any]

   Spawn single node, multi-rank function.
   Uses localhost and free port to communicate.

   :param world_size: number of processes
   :param backend: backend to use. for example, "nccl", "gloo", etc
   :param test_method: callable to spawn. first 3 arguments are rank, world_size and mp output dict
   :param test_method_args: args for the test method
   :param test_method_kwargs: kwargs for the test method

   :returns: A list, l, where l[i] is the return value of test_method on rank i


.. py:function:: _init_pg_and_rank_and_launch_test(rank: int, pg_setup_params: PGConfig, mp_output_dict: dict[int, object], test_method: callable, args: list[object], kwargs: dict[str, object]) -> None

