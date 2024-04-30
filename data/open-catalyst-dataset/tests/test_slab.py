import random

import numpy as np

from ocdata.core import Bulk, Slab


class TestSlab:
    def test_slab_init_from_id(self):
        bulk = Bulk(bulk_id_from_db=0)
        slabs = Slab.from_bulk_get_all_slabs(bulk)

        assert slabs[0].atoms.get_chemical_formula() == "Re48"
        assert slabs[0].millers == (1, 1, 1)
        assert slabs[0].shift == 0.0

    def test_slab_init_from_specific_millers(self):
        bulk = Bulk(bulk_src_id_from_db="mp-30")
        slabs = Slab.from_bulk_get_specific_millers(
            specific_millers=(1, 1, 1), bulk=bulk
        )

        assert len(slabs) == 1
        assert slabs[0].millers == (1, 1, 1)

    def test_slab_init_random(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk(bulk_id_from_db=100)
        slab = Slab.from_bulk_get_random_slab(bulk)

        assert slab.atoms.get_chemical_formula() == "Sn48"
        assert slab.millers == (2, 1, 0)
        assert slab.shift == 0.0833333333333334
