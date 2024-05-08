:py:mod:`fairchem.core.tests.datasets.test_ase_lmdb`
====================================================

.. py:module:: fairchem.core.tests.datasets.test_ase_lmdb


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.tests.datasets.test_ase_lmdb.generate_random_structure
   fairchem.core.tests.datasets.test_ase_lmdb.ase_lmbd_path
   fairchem.core.tests.datasets.test_ase_lmdb.test_aselmdb_write
   fairchem.core.tests.datasets.test_ase_lmdb.test_aselmdb_count
   fairchem.core.tests.datasets.test_ase_lmdb.test_aselmdb_delete
   fairchem.core.tests.datasets.test_ase_lmdb.test_aselmdb_randomreads
   fairchem.core.tests.datasets.test_ase_lmdb.test_aselmdb_constraintread
   fairchem.core.tests.datasets.test_ase_lmdb.test_update_keyvalue_pair
   fairchem.core.tests.datasets.test_ase_lmdb.test_update_atoms
   fairchem.core.tests.datasets.test_ase_lmdb.test_metadata



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.tests.datasets.test_ase_lmdb.N_WRITES
   fairchem.core.tests.datasets.test_ase_lmdb.N_READS
   fairchem.core.tests.datasets.test_ase_lmdb.test_structures


.. py:data:: N_WRITES
   :value: 100

   

.. py:data:: N_READS
   :value: 200

   

.. py:data:: test_structures

   

.. py:function:: generate_random_structure()


.. py:function:: ase_lmbd_path(tmp_path_factory)


.. py:function:: test_aselmdb_write(ase_lmbd_path) -> None


.. py:function:: test_aselmdb_count(ase_lmbd_path) -> None


.. py:function:: test_aselmdb_delete(ase_lmbd_path) -> None


.. py:function:: test_aselmdb_randomreads(ase_lmbd_path) -> None


.. py:function:: test_aselmdb_constraintread(ase_lmbd_path) -> None


.. py:function:: test_update_keyvalue_pair(ase_lmbd_path) -> None


.. py:function:: test_update_atoms(ase_lmbd_path) -> None


.. py:function:: test_metadata(ase_lmbd_path) -> None


