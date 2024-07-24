from __future__ import annotations

import pickle

import ase
import ase.units as units
from fairchem.data.oc.databases.pkls import SOLVENT_PKL_PATH


class Solvent:
    """
    Initializes a solvent object in one of 2 ways:
    - Directly pass in an ase.Atoms object.
    - Pass in index of solvent to select from solvent database.

    Arguments
    ---------
    solvent_atoms: ase.Atoms
        Solvent molecule
    solvent_id_from_db: int
        Index of solvent to select.
    solvent_db_path: str
        Path to solvent database.
    solvent_density: float
        Desired solvent density to use. If not specified, the default is used
        from the solvent databases.
    """

    def __init__(
        self,
        solvent_atoms: ase.Atoms = None,
        solvent_id_from_db: int | None = None,
        solvent_db_path: str | None = SOLVENT_PKL_PATH,
        solvent_density: float | None = None,
    ):
        self.solvent_id_from_db = solvent_id_from_db
        self.solvent_db_path = solvent_db_path
        self.solvent_density = solvent_density

        if solvent_atoms is not None:
            self.atoms = solvent_atoms.copy()
            self.name = str(self.atoms.symbols)
        elif solvent_id_from_db is not None:
            with open(solvent_db_path, "rb") as fp:
                solvent_db = pickle.load(fp)
            self._load_solvent(solvent_db[solvent_id_from_db])

        self.molar_mass = sum(self.atoms.get_masses())

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return self.name

    def _load_solvent(self, solvent: dict) -> None:
        """
        Saves the fields from an adsorbate stored in a database. Fields added
        after the first revision are conditionally added for backwards
        compatibility with older database files.
        """
        self.atoms = solvent["atoms"]
        self.name = solvent["name"]
        # use the default density if one is not specified
        self.density = (
            solvent["density"] if not self.solvent_density else self.solvent_density
        )

    @property
    def molecules_per_volume(self):
        """
        Convert the solvent density in g/cm3 to the number of molecules per
        angstrom cubed of volume.
        """
        # molecules/mol * grams/cm3 / (1e24 A^3/cm^3 * g/mol)
        return units._Nav * self.density / (1e24 * self.molar_mass)
