import random

import ase
import numpy as np

from fairchem.data.oc.core import Adsorbate


_test_db = {
    0: (ase.Atoms(symbols="H", pbc="False"), "*H", np.array([0]), "rxn_1"),
    1: (ase.Atoms(symbols="C", pbc="False"), "*C", np.array([0]), "rxn_2"),
}

# Used to test backwards compatability with old database formats
_test_db_old = {
    0: (ase.Atoms(symbols="H", pbc="False"), "*H", np.array([0])),
}


class TestAdsorbate:
    def test_adsorbate_init_from_id(self):
        adsorbate = Adsorbate(adsorbate_id_from_db=0)
        assert adsorbate.atoms.get_chemical_formula() == "O"
        assert adsorbate.smiles == "*O"
        assert adsorbate.adsorbate_id_from_db == 0

    def test_adsorbate_init_from_smiles(self):
        adsorbate = Adsorbate(adsorbate_smiles_from_db="*H")
        assert adsorbate.atoms.get_chemical_formula() == "H"
        assert adsorbate.adsorbate_id_from_db == 1

    def test_adsorbate_init_random(self):
        random.seed(1)
        np.random.seed(1)

        adsorbate = Adsorbate()
        assert adsorbate.atoms.get_chemical_formula() == "C2H3O"
        assert adsorbate.smiles == "*COHCH2"

    def test_adsorbate_init_from_id_with_db(self):
        adsorbate = Adsorbate(adsorbate_id_from_db=1, adsorbate_db=_test_db)
        assert adsorbate.atoms.get_chemical_formula() == "C"

    def test_adsorbate_init_from_smiles_with_db(self):
        adsorbate = Adsorbate(adsorbate_smiles_from_db="*C", adsorbate_db=_test_db)
        assert adsorbate.atoms.get_chemical_formula() == "C"

    def test_adsorbate_init_random_with_db(self):
        random.seed(1)
        np.random.seed(1)

        adsorbate = Adsorbate(adsorbate_db=_test_db)
        assert adsorbate.atoms.get_chemical_formula() == "C"

    def test_adsorbate_init_reaction_string(self):
        adsorbate = Adsorbate(adsorbate_id_from_db=0, adsorbate_db=_test_db)
        assert adsorbate.reaction_string == "rxn_1"

    def test_adsorbate_init_reaction_string_with_old_db(self):
        adsorbate = Adsorbate(adsorbate_id_from_db=0, adsorbate_db=_test_db_old)
        assert not hasattr(adsorbate, "reaction_string")
