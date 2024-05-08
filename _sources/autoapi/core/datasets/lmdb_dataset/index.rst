:py:mod:`core.datasets.lmdb_dataset`
====================================

.. py:module:: core.datasets.lmdb_dataset

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.datasets.lmdb_dataset.LmdbDataset
   core.datasets.lmdb_dataset.SinglePointLmdbDataset
   core.datasets.lmdb_dataset.TrajectoryLmdbDataset



Functions
~~~~~~~~~

.. autoapisummary::

   core.datasets.lmdb_dataset.data_list_collater



Attributes
~~~~~~~~~~

.. autoapisummary::

   core.datasets.lmdb_dataset.T_co


.. py:data:: T_co

   

.. py:class:: LmdbDataset(config)


   Bases: :py:obj:`torch.utils.data.Dataset`\ [\ :py:obj:`T_co`\ ]

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:attribute:: metadata_path
      :type: pathlib.Path

      

   .. py:attribute:: sharded
      :type: bool

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

   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx: int) -> T_co


   .. py:method:: connect_db(lmdb_path: pathlib.Path | None = None) -> lmdb.Environment


   .. py:method:: close_db() -> None


   .. py:method:: get_metadata(num_samples: int = 100)



.. py:class:: SinglePointLmdbDataset(config, transform=None)


   Bases: :py:obj:`LmdbDataset`\ [\ :py:obj:`torch_geometric.data.data.BaseData`\ ]

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


.. py:class:: TrajectoryLmdbDataset(config, transform=None)


   Bases: :py:obj:`LmdbDataset`\ [\ :py:obj:`torch_geometric.data.data.BaseData`\ ]

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


.. py:function:: data_list_collater(data_list: list[torch_geometric.data.data.BaseData], otf_graph: bool = False) -> torch_geometric.data.data.BaseData


