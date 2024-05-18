import os
import pickle
import random

import ase
import numpy as np
import pytest

from fairchem.data.oc.core import Bulk


@pytest.fixture(scope="class")
def load_bulk(request):
    cwd = os.getcwd()
    request.cls.idx = 0

    request.cls.precomputed_path = os.path.join(cwd, str(request.cls.idx) + ".pkl")
    request.cls.bulk = Bulk(bulk_id_from_db=request.cls.idx)


_test_db = [
    {
        "atoms": ase.Atoms(symbols="H", pbc=False),
        "src_id": "test_id_1",
        "bulk_sampling_str": "test_1",
    },
    {
        "atoms": ase.Atoms(symbols="C", pbc=False),
        "src_id": "test_id_2",
        "bulk_sampling_str": "test_2",
    },
]


@pytest.mark.usefixtures("load_bulk")
class TestBulk:
    def test_bulk_init_from_id(self):
        bulk = Bulk(bulk_id_from_db=self.idx)
        assert bulk.atoms.get_chemical_formula() == "Re2"
        assert bulk.bulk_id_from_db == self.idx

    def test_bulk_init_from_src_id(self):
        bulk = Bulk(bulk_src_id_from_db="mp-30")
        assert bulk.atoms.get_chemical_formula() == "Cu"
        assert bulk.src_id == "mp-30"

    def test_bulk_init_random(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk()
        assert bulk.atoms.get_chemical_formula() == "IrSn2"

    def test_bulk_init_from_id_with_db(self):
        bulk = Bulk(bulk_id_from_db=1, bulk_db=_test_db)
        assert bulk.atoms.get_chemical_formula() == "C"

    def test_bulk_init_from_src_id_with_db(self):
        bulk = Bulk(bulk_src_id_from_db="test_id_2", bulk_db=_test_db)
        assert bulk.atoms.get_chemical_formula() == "C"

    def test_bulk_init_random_with_db(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk(bulk_db=_test_db)
        assert bulk.atoms.get_chemical_formula() == "C"

    def test_unique_slab_enumeration(self):
        slabs = self.bulk.get_slabs()

        seen = []
        for slab in slabs:
            assert slab not in seen
            seen.append(slab)

        # pymatgen bug see https://github.com/materialsproject/pymatgen/issues/3747
        if len(slabs) == 15:
            pytest.xfail(
                f"Number of generated slabs {len(slabs)} is off due to pymatgen bug!"
            )
        assert len(slabs) == 14

        with open(self.precomputed_path, "wb") as f:
            pickle.dump(slabs, f)

    def test_precomputed_slab(self):
        precomputed_slabs_dir = os.path.dirname(self.precomputed_path)

        precomputed_slabs = self.bulk.get_slabs(
            precomputed_slabs_dir=precomputed_slabs_dir
        )

        if len(precomputed_slabs) == 15:
            pytest.xfail(
                f"Number of generated slabs {len(precomputed_slabs)} is off due to pymatgen bug!"
            )

        assert len(precomputed_slabs) == 14

        slabs = self.bulk.get_slabs()
        assert precomputed_slabs[0] == slabs[0]

        os.remove(self.precomputed_path)

    def test_slab_miller_enumeration(self):
        slabs_max_miller_1 = self.bulk.get_slabs(max_miller=1)
        assert self.get_max_miller(slabs_max_miller_1) == 1
        slabs_max_miller_2 = self.bulk.get_slabs(max_miller=2)
        assert self.get_max_miller(slabs_max_miller_2) == 2
        slabs_max_miller_3 = self.bulk.get_slabs(max_miller=3)
        assert self.get_max_miller(slabs_max_miller_3) == 3

    def get_max_miller(self, slabs):
        max_miller = 0
        for slab in slabs:
            millers = slab.millers
            max_miller = max(max_miller, max(millers))

        return max_miller
