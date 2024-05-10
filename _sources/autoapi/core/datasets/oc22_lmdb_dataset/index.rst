:py:mod:`core.datasets.oc22_lmdb_dataset`
=========================================

.. py:module:: core.datasets.oc22_lmdb_dataset

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.datasets.oc22_lmdb_dataset.OC22LmdbDataset




.. py:class:: OC22LmdbDataset(config, transform=None)


   Bases: :py:obj:`torch.utils.data.Dataset`

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
   :param transform: Data transform function.
                     (default: :obj:`None`)
   :type transform: callable, optional

   .. py:method:: __len__() -> int


   .. py:method:: __getitem__(idx)


   .. py:method:: connect_db(lmdb_path=None)


   .. py:method:: close_db() -> None



