import random

import numpy as np

from ocdata.core import Adsorbate


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
