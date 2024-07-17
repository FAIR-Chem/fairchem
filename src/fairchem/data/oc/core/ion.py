from __future__ import annotations

import pickle

import ase
import ase.units as units
import numpy as np
from fairchem.data.oc.databases.pkls import ION_PKL_PATH


class Ion:
    """
    Initializes an ion object in one of 2 ways:
    - Directly pass in an ase.Atoms object.
    - Pass in index of ion to select from ion database.

    Arguments
    ---------
    ion_atoms: ase.Atoms
        ion structure.
    ion_id_from_db: int
        Index of ion to select.
    ion_db_path: str
        Path to ion database.
    """

    def __init__(
        self,
        ion_atoms: ase.Atoms = None,
        ion_id_from_db: int | None = None,
        ion_db_path: str = ION_PKL_PATH,
    ):
        self.ion_id_from_db = ion_id_from_db
        self.ion_db_path = ion_db_path

        if ion_atoms is not None:
            self.atoms = ion_atoms.copy()
            self.name = str(self.atoms.symbols)
        else:
            with open(ion_db_path, "rb") as fp:
                ion_db = pickle.load(fp)
            if ion_id_from_db is not None:
                self._load_ion(ion_db[ion_id_from_db])
            else:
                self.ion_id_from_db = np.random.randint(len(ion_db))
                self._load_ion(ion_db[self.ion_id_from_db])

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return self.name

    def _load_ion(self, ion: dict) -> None:
        """
        Saves the fields from an ion stored in a database. Fields added
        after the first revision are conditionally added for backwards
        compatibility with older database files.
        """
        self.atoms = ion["atoms"]
        self.name = ion["name"]
        self.charge = ion["charge"]

    def get_ion_concentration(self, volume):
        """
        Compute the ion concentration units of M, given a volume in units of
        Angstrom^3.
        """
        return 1e27 / (units._Nav * volume)
