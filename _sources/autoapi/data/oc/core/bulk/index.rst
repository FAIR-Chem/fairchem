data.oc.core.bulk
=================

.. py:module:: data.oc.core.bulk


Classes
-------

.. autoapisummary::

   data.oc.core.bulk.Bulk


Module Contents
---------------

.. py:class:: Bulk(bulk_atoms: ase.Atoms = None, bulk_id_from_db: int | None = None, bulk_src_id_from_db: str | None = None, bulk_db_path: str = BULK_PKL_PATH, bulk_db: list[dict[str, Any]] | None = None)

   Initializes a bulk object in one of 4 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of bulk to select from bulk database.
   - Pass in the src_id of the bulk to select from the bulk database.
   - Randomly sample a bulk from bulk database if no other option is passed.

   :param bulk_atoms: Bulk structure.
   :type bulk_atoms: ase.Atoms
   :param bulk_id_from_db: Index of bulk in database pkl to select.
   :type bulk_id_from_db: int
   :param bulk_src_id_from_db: Src id of bulk to select (e.g. "mp-30").
   :type bulk_src_id_from_db: int
   :param bulk_db_path: Path to bulk database.
   :type bulk_db_path: str
   :param bulk_db: Already-loaded database.
   :type bulk_db: List[Dict[str, Any]]


   .. py:method:: _get_bulk_from_random(bulk_db)


   .. py:method:: set_source_dataset_id(src_id: str)


   .. py:method:: set_bulk_id_from_db(bulk_id_from_db: int)


   .. py:method:: get_slabs(max_miller=2, precomputed_slabs_dir=None)

      Returns a list of possible slabs for this bulk instance.



   .. py:method:: __len__()


   .. py:method:: __str__()

      Return str(self).



   .. py:method:: __repr__()

      Return repr(self).



   .. py:method:: __eq__(other) -> bool

      Return self==value.



