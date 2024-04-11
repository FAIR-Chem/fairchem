import random

import numpy as np
import pytest
from ase.data import covalent_radii
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from ocdata.core.adsorbate_slab_config import get_interstitial_distances
from ocdata.databases.pkls import ADSORBATES_PKL_PATH, BULK_PKL_PATH


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)
    request.cls.adsorbate = Adsorbate(adsorbate_id_from_db=80)


@pytest.mark.usefixtures("load_data")
class TestAdslab:
    def test_adslab_init(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100)
        assert (
            len(adslab.atoms_list) == 100
        ), f"Insufficient number of structures. Expected 100, got {len(adslab.atoms_list)}"

        sites = ["%.04f_%.04f_%.04f" % (i[0], i[1], i[2]) for i in adslab.sites]
        assert (
            len(set(sites)) == 100
        ), f"Insufficient number of sites. Expected 100, got {len(set(sites))}"

        assert np.all(
            np.isclose(
                adslab.atoms_list[0].get_positions().mean(0),
                np.array([6.2668884, 4.22961421, 16.47458617]),
            )
        )
        assert np.all(
            np.isclose(
                adslab.atoms_list[1].get_positions().mean(0),
                np.array([6.1967168, 4.73603662, 16.46990669]),
            )
        )

    def test_num_augmentations_per_site(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=1, num_augmentations_per_site=100
        )
        assert len(adslab.atoms_list) == 100

        sites = ["%.04f_%.04f_%.04f" % (i[0], i[1], i[2]) for i in adslab.sites]
        assert len(set(sites)) == 1

    def test_placement_overlap(self):
        """
        Test that the adsorbate does not overlap with the slab.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, interstitial_gap=0.1
        )
        assert len(adslab.atoms_list) == 100

        min_distance_close = []
        for i in adslab.atoms_list:
            min_distance_close.append(
                np.isclose(min(get_interstitial_distances(i)), 0.1)
            )
        assert all(min_distance_close)

        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, interstitial_gap=0.5
        )
        min_distance_close = []
        for i in adslab.atoms_list:
            min_distance_close.append(
                np.isclose(min(get_interstitial_distances(i)), 0.5)
            )
        assert all(min_distance_close)

    def test_is_adsorbate_com_on_normal(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        normal = np.cross(slab.atoms.cell[0], slab.atoms.cell[1])
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100, mode="random")
        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)

        cp_test = []
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            adsorbate_com = adsorbate_atoms.get_center_of_mass()
            cp = np.cross(normal, adsorbate_com - site)
            cp_test.append(cp)
            assert np.isclose(cp_test, 0).all()

    def test_is_adsorbate_binding_atom_on_normal(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        normal = np.cross(slab.atoms.cell[0], slab.atoms.cell[1])
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, mode="heuristic"
        )
        binding_idx = self.adsorbate.binding_indices[0]
        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)

        cp_test = []
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            binding_atom = adsorbate_atoms[binding_idx].position
            cp = np.cross(normal, binding_atom - site)
            cp_test.append(cp)
            assert np.isclose(cp_test, 0).all()
