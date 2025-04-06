from __future__ import annotations

import random

import numpy as np
import pytest
from fairchem.data.oc.core import Adsorbate, Bulk, InterfaceConfig, Ion, Slab, Solvent


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)

    adsorbate_idx = [0, 1]
    adsorbates = [Adsorbate(adsorbate_id_from_db=i) for i in adsorbate_idx]

    solvent = Solvent(solvent_id_from_db=0)
    ions = [Ion(ion_id_from_db=3)]

    request.cls.adsorbates = adsorbates
    request.cls.solvent = solvent
    request.cls.ions = ions
    request.cls.vacuum = 15
    request.cls.solvent_depth = 10


@pytest.mark.usefixtures("load_data")
class TestInterface:
    def test_num_configurations(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslabs = InterfaceConfig(
            slab,
            self.adsorbates,
            self.solvent,
            self.ions,
            vacuum_size=self.vacuum,
            solvent_depth=self.solvent_depth,
            num_configurations=10,
        )
        assert len(adslabs.atoms_list) == 10
        assert len(adslabs.metadata_list) == 10

    def test_solvent_density(self):
        """
        Test that the number of solvent + ion molecules inside the environment
        is consistent with the specified density.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslabs = InterfaceConfig(
            slab,
            self.adsorbates,
            self.solvent,
            self.ions,
            vacuum_size=self.vacuum,
            solvent_depth=self.solvent_depth,
            num_configurations=10,
        )

        for atoms, metadata in zip(adslabs.atoms_list, adslabs.metadata_list):
            volume = metadata["solvent_volume"]
            n_solvent_mols = int(volume * self.solvent.molecules_per_volume)
            n_solvent_atoms = n_solvent_mols * len(self.solvent.atoms)
            n_ions = len(self.ions)

            solvent_ions_atoms = atoms[atoms.get_tags() == 3]
            assert len(solvent_ions_atoms) == n_solvent_atoms + n_ions
