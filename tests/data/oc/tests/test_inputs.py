import random

import numpy as np
import pytest

from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from fairchem.data.oc.utils.vasp import VASP_FLAGS, _clean_up_inputs


@pytest.fixture(scope="class")
def load_data(request):
    random.seed(1)
    np.random.seed(1)

    bulk_sample_1 = Bulk(bulk_id_from_db=24)
    slab_sample_1 = Slab.from_bulk_get_random_slab(bulk_sample_1)
    adsorbate_sample_1 = Adsorbate(adsorbate_id_from_db=10)

    bulk_sample_2 = Bulk(bulk_id_from_db=100)
    slab_sample_2 = Slab.from_bulk_get_random_slab(bulk_sample_2)
    adsorbate_sample_2 = Adsorbate(adsorbate_id_from_db=2)

    request.cls.adslab1 = AdsorbateSlabConfig(
        slab_sample_1, adsorbate_sample_1, num_sites=100
    )
    request.cls.adslab2 = AdsorbateSlabConfig(
        slab_sample_2, adsorbate_sample_2, num_sites=100
    )

    ALT_VASP_FLAGS = VASP_FLAGS.copy()
    ALT_VASP_FLAGS["nsw"] = 0
    ALT_VASP_FLAGS["laechg"] = False
    ALT_VASP_FLAGS["ncore"] = 1
    request.cls.alt_flags = ALT_VASP_FLAGS


@pytest.mark.usefixtures("load_data")
class TestVasp:
    def test_cleanup(self):
        atoms = self.adslab1.atoms_list[0]
        atoms1, flags1 = _clean_up_inputs(atoms, VASP_FLAGS)

        # Check that kpts are computed and added to the flags
        assert "kpts" in flags1
        # Check that kpts weren't added to the original flags
        assert "kpts" not in VASP_FLAGS

        atoms2, flags2 = _clean_up_inputs(atoms, self.alt_flags)

        assert atoms1 == atoms2
        assert flags2 != flags1

    def test_unique_kpts(self):
        atoms1 = self.adslab1.atoms_list[0]
        atoms2 = self.adslab2.atoms_list[0]

        _, flags1 = _clean_up_inputs(atoms1, VASP_FLAGS)
        _, flags2 = _clean_up_inputs(atoms2, VASP_FLAGS)

        assert flags1 != flags2
