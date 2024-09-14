data.oc.core.adsorbate
======================

.. py:module:: data.oc.core.adsorbate


Classes
-------

.. autoapisummary::

   data.oc.core.adsorbate.Adsorbate


Functions
---------

.. autoapisummary::

   data.oc.core.adsorbate.randomly_rotate_adsorbate


Module Contents
---------------

.. py:class:: Adsorbate(adsorbate_atoms: ase.Atoms = None, adsorbate_id_from_db: int | None = None, adsorbate_smiles_from_db: str | None = None, adsorbate_db_path: str = ADSORBATE_PKL_PATH, adsorbate_db: dict[int, tuple[Any, Ellipsis]] | None = None, adsorbate_binding_indices: list | None = None)

   Initializes an adsorbate object in one of 4 ways:
   - Directly pass in an ase.Atoms object.
       For this, you should also provide the index of the binding atom.
   - Pass in index of adsorbate to select from adsorbate database.
   - Pass in the SMILES string of the adsorbate to select from the database.
   - Randomly sample an adsorbate from the adsorbate database.

   :param adsorbate_atoms: Adsorbate structure.
   :type adsorbate_atoms: ase.Atoms
   :param adsorbate_id_from_db: Index of adsorbate to select.
   :type adsorbate_id_from_db: int
   :param adsorbate_smiles_from_db: A SMILES string of the desired adsorbate.
   :type adsorbate_smiles_from_db: str
   :param adsorbate_db_path: Path to adsorbate database.
   :type adsorbate_db_path: str
   :param adsorbate_binding_indices: The index/indices of the adsorbate atoms which are expected to bind.
   :type adsorbate_binding_indices: list


   .. py:attribute:: adsorbate_id_from_db


   .. py:attribute:: adsorbate_db_path


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: _get_adsorbate_from_random(adsorbate_db)


   .. py:method:: _load_adsorbate(adsorbate: tuple[Any, Ellipsis]) -> None

      Saves the fields from an adsorbate stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



.. py:function:: randomly_rotate_adsorbate(adsorbate_atoms: ase.Atoms, mode: str = 'random', binding_idx: int | None = None)

