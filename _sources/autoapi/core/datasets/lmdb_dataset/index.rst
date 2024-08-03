core.datasets.lmdb_dataset
==========================

.. py:module:: core.datasets.lmdb_dataset

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.datasets.lmdb_dataset.T_co


Classes
-------

.. autoapisummary::

   core.datasets.lmdb_dataset.LmdbDataset


Functions
---------

.. autoapisummary::

   core.datasets.lmdb_dataset.data_list_collater


Module Contents
---------------

.. py:data:: T_co

.. py:class:: LmdbDataset(config)

   Bases: :py:obj:`fairchem.core.datasets.base_dataset.BaseDataset`


   Base Dataset class for all OCP datasets.


   .. py:attribute:: sharded
      :type:  bool

      Dataset class to load from LMDB files containing relaxation
      trajectories or single point computations.
      Useful for Structure to Energy & Force (S2EF), Initial State to
      Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
      The keys in the LMDB must be integers (stored as ascii objects) starting
      from 0 through the length of the LMDB. For historical reasons any key named
      "length" is ignored since that was used to infer length of many lmdbs in the same
      folder, but lmdb lengths are now calculated directly from the number of keys.
      :param config: Dataset configuration
      :type config: dict


   .. py:attribute:: path


   .. py:attribute:: key_mapping


   .. py:attribute:: transforms


   .. py:method:: __getitem__(idx: int) -> T_co


   .. py:method:: connect_db(lmdb_path: pathlib.Path | None = None) -> lmdb.Environment


   .. py:method:: __del__()


   .. py:method:: sample_property_metadata(num_samples: int = 100)


.. py:function:: data_list_collater(data_list: list[torch_geometric.data.data.BaseData], otf_graph: bool = False) -> torch_geometric.data.data.BaseData

