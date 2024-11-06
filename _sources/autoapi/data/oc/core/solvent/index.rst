data.oc.core.solvent
====================

.. py:module:: data.oc.core.solvent


Classes
-------

.. autoapisummary::

   data.oc.core.solvent.Solvent


Module Contents
---------------

.. py:class:: Solvent(solvent_atoms: ase.Atoms = None, solvent_id_from_db: int | None = None, solvent_db_path: str | None = SOLVENT_PKL_PATH, solvent_density: float | None = None)

   Initializes a solvent object in one of 2 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of solvent to select from solvent database.

   :param solvent_atoms: Solvent molecule
   :type solvent_atoms: ase.Atoms
   :param solvent_id_from_db: Index of solvent to select.
   :type solvent_id_from_db: int
   :param solvent_db_path: Path to solvent database.
   :type solvent_db_path: str
   :param solvent_density: Desired solvent density to use. If not specified, the default is used
                           from the solvent databases.
   :type solvent_density: float


   .. py:attribute:: solvent_id_from_db


   .. py:attribute:: solvent_db_path


   .. py:attribute:: solvent_density


   .. py:attribute:: molar_mass


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: _load_solvent(solvent: dict) -> None

      Saves the fields from an adsorbate stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



   .. py:property:: molecules_per_volume

      Convert the solvent density in g/cm3 to the number of molecules per
      angstrom cubed of volume.


