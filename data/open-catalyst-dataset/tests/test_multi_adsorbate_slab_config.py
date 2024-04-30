import random

import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.core import Adsorbate, Bulk, MultipleAdsorbateSlabConfig, Slab
from ocdata.core.adsorbate_slab_config import get_interstitial_distances
from ocdata.databases.pkls import ADSORBATES_PKL_PATH, BULK_PKL_PATH


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)

    adsorbate_idx = [0, 1, 3, 80, 5]
    adsorbates = [Adsorbate(adsorbate_id_from_db=i) for i in adsorbate_idx]

    request.cls.adsorbates = adsorbates


@pytest.mark.usefixtures("load_data")
class TestMultiAdslab:
    def test_num_configurations(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslabs = MultipleAdsorbateSlabConfig(
            slab, self.adsorbates, num_configurations=100
        )
        assert len(adslabs.atoms_list) == 100
        assert len(adslabs.metadata_list) == 100

    def test_adsorbate_indices(self):
        """
        Test that the adsorbate indices correspond to the unique adsorbates.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslabs = MultipleAdsorbateSlabConfig(
            slab, self.adsorbates, num_configurations=10
        )

        for atoms, metadata in zip(adslabs.atoms_list, adslabs.metadata_list):
            atomic_numbers = np.array(atoms.get_chemical_symbols())

            for adsorbate, ads_metadata in zip(self.adsorbates, metadata):
                expected_adsorbate_atomic_numbers = (
                    adsorbate.atoms.get_chemical_symbols()
                )
                adsorbate_atomic_numbers = atomic_numbers[
                    ads_metadata["adsorbate_indices"]
                ]

                assert len(adsorbate_atomic_numbers) == len(
                    expected_adsorbate_atomic_numbers
                )
                assert set(adsorbate_atomic_numbers) == set(
                    expected_adsorbate_atomic_numbers
                )

    def test_placement_overlap(self):
        """
        Test that the adsorbate sites do not overlap with each other.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslabs = MultipleAdsorbateSlabConfig(
            slab, self.adsorbates, interstitial_gap=0.1, num_configurations=100
        )

        for atoms, metadata in zip(adslabs.atoms_list, adslabs.metadata_list):
            positions = []
            atomic_numbers = []

            for ads_placement in metadata:
                adsorbate = ads_placement["adsorbate"]
                adsorbate_binding_atom = adsorbate.atoms.get_chemical_symbols()[
                    adsorbate.binding_indices[0]
                ]
                atomic_numbers.append(adsorbate_binding_atom)
                positions.append(ads_placement["site"])

            pseudo_atoms = Atoms(
                positions=positions, symbols=atomic_numbers, cell=atoms.get_cell()
            )

            raw_distances = pseudo_atoms.get_all_distances(mic=True)

            covalent_radii_correction = (
                covalent_radii[pseudo_atoms.get_atomic_numbers().reshape(1, -1)]
                + covalent_radii[pseudo_atoms.get_atomic_numbers().reshape(-1, 1)]
                + 0.1
            )

            adjusted_distances = raw_distances - covalent_radii_correction

            # Diagonal elements correspond to same atom, so set to large number
            # to satisfy test.
            np.fill_diagonal(adjusted_distances, 1e10)

            assert np.all(adjusted_distances > 0)
