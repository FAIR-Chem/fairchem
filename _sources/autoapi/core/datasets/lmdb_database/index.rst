core.datasets.lmdb_database
===========================

.. py:module:: core.datasets.lmdb_database

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is modified from the ASE db json backend
   and is thus licensed under the corresponding LGPL2.1 license

   The ASE notice for the LGPL2.1 license is available here:
   https://gitlab.com/ase/ase/-/blob/master/LICENSE



Attributes
----------

.. autoapisummary::

   core.datasets.lmdb_database.RESERVED_KEYS


Classes
-------

.. autoapisummary::

   core.datasets.lmdb_database.LMDBDatabase


Module Contents
---------------

.. py:data:: RESERVED_KEYS
   :value: ['nextid', 'metadata', 'deleted_ids']


.. py:class:: LMDBDatabase(filename: str | pathlib.Path | None = None, create_indices: bool = True, use_lock_file: bool = False, serial: bool = False, readonly: bool = False, *args, **kwargs)

   Bases: :py:obj:`ase.db.core.Database`


   Base class for all databases.


   .. py:attribute:: readonly


   .. py:attribute:: ids
      :value: []



   .. py:attribute:: deleted_ids
      :value: []



   .. py:method:: __enter__() -> typing_extensions.Self


   .. py:method:: __exit__(exc_type, exc_value, tb) -> None


   .. py:method:: close() -> None


   .. py:method:: _write(atoms: ase.Atoms | ase.db.row.AtomsRow, key_value_pairs: dict, data: dict | None, idx: int | None = None) -> None


   .. py:method:: _update(idx: int, key_value_pairs: dict | None = None, data: dict | None = None)


   .. py:method:: _write_deleted_ids()


   .. py:method:: delete(ids: list[int]) -> None

      Delete rows.



   .. py:method:: _get_row(idx: int, include_data: bool = True)


   .. py:method:: _get_row_by_index(index: int, include_data: bool = True)

      Auxiliary function to get the ith entry, rather than a specific id



   .. py:method:: _select(keys, cmps: list[tuple[str, str, str]], explain: bool = False, verbosity: int = 0, limit: int | None = None, offset: int = 0, sort: str | None = None, include_data: bool = True, columns: str = 'all')


   .. py:property:: metadata

      Load the metadata from the DB if present


   .. py:property:: _nextid

      Get the id of the next row to be written


   .. py:method:: count(selection=None, **kwargs) -> int

      Count rows.

      See the select() method for the selection syntax.  Use db.count() or
      len(db) to count all rows.



   .. py:method:: _load_ids() -> None

      Load ids from the DB

      Since ASE db ids are mostly 1-N integers, but can be missing entries
      if ids have been deleted. To save space and operating under the assumption
      that there will probably not be many deletions in most OCP datasets,
      we just store the deleted ids.



