"""
Script for updating ase pkl and db files from v3.19 to v3.21.
Run it with ase v3.19.
"""
import pickle

import ase.io
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from tqdm import tqdm


# Monkey patch fix
def pbc_patch(self):
    return self.cell._pbc


def set_pbc_patch(self, pbc):
    self.cell._pbc = pbc
    self._pbc = pbc


Atoms.pbc = property(pbc_patch)
Atoms.set_pbc = set_pbc_patch


def update_pkls():
    data = pickle.load(
        open(
            "ocdata/databases/pkls/adsorbates.pkl",
            "rb",
        )
    )
    for idx in data:
        pbc = data[idx][0].cell._pbc
        data[idx][0]._pbc = pbc
    with open(
        "ocdata/databases/pkls/adsorbates_new.pkl",
        "wb",
    ) as f:
        pickle.dump(data, f)

    data = pickle.load(
        open(
            "ocdata/databases/pkls/bulks.pkl",
            "rb",
        )
    )

    bulks = []
    for info in tqdm(data):
        atoms, bulk_id, _, _ = info
        pbc = atoms.cell._pbc
        atoms._pbc = pbc

        if hasattr(atoms, "calc"):
            temp_energy = atoms.get_potential_energy()
            temp_forces = atoms.get_forces()
            temp_calc = SPC(atoms=atoms, energy=temp_energy, forces=temp_forces)
            temp_calc.implemented_properties = ["energy", "forces"]
            atoms.set_calculator(temp_calc)

        bulks.append((atoms, bulk_id))
    with open(
        "ocdata/databases/pkls/bulks_new.pkl",
        "wb",
    ) as f:
        pickle.dump(bulks, f)


def update_dbs():
    for db_name in ["adsorbates", "bulks"]:
        db = ase.io.read(
            f"ocdata/databases/ase/{db_name}.db",
            ":",
        )
        new_data = []
        for atoms in tqdm(db):
            pbc = atoms.cell._pbc
            atoms._pbc = pbc

            if hasattr(atoms, "calc"):
                temp_energy = atoms.get_potential_energy()
                temp_forces = atoms.get_forces()
                temp_calc = SPC(atoms=atoms, energy=temp_energy, forces=temp_forces)
                temp_calc.implemented_properties = ["energy", "forces"]
                atoms.set_calculator(temp_calc)
            new_data.append(atoms)

        ase.io.write(
            f"ocdata/databases/ase/{db_name}_new.db",
            new_data,
        )


if __name__ == "__main__":
    update_pkls()
    update_dbs()
