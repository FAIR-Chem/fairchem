import os

import ase
import numpy as np
import pytest
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from ocpmodels.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def atoms_to_graphs_internals(request):
    atoms = read(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_atoms.json"
        ),
        index=0,
        format="json",
    )
    test_object = AtomsToGraphs(
        max_neigh=12,
        radius=6,
        dummy_distance=7,
        dummy_index=-1,
        r_energy=True,
        r_forces=True,
        r_distances=True,
    )
    request.cls.atg = test_object
    request.cls.atoms = atoms


@pytest.mark.usefixtures("atoms_to_graphs_internals")
class TestAtomsToGraphs:
    def test_gen_neighbors_pymatgen(self):
        # call the internal function
        split_n_index, split_n_distances = self.atg._get_neighbors_pymatgen(
            self.atoms
        )
        act_struct = AseAtomsAdaptor.get_structure(self.atoms)
        # use the old pymatgen method to get distance and indicies
        act_neigh = act_struct.get_all_neighbors(r=self.atg.radius)
        # randomly chose to check the neighbors and index of atom 4
        _, distances, indices, _ = zip(*act_neigh[4])
        act_dist = np.sort(distances)
        act_index = np.sort(indices)
        test_dist = np.sort(split_n_distances[4])
        test_index = np.sort(split_n_index[4])
        # check that the distance and neighbor index values are correct
        np.testing.assert_allclose(act_dist, test_dist)
        np.testing.assert_array_equal(act_index, test_index)
        # check for the correct length
        # the number of neighbors varies so hard to test that
        act_len = len(self.atoms)
        len_dist = len(split_n_distances)
        len_index = len(split_n_index)
        np.testing.assert_equal(act_len, len_dist)
        np.testing.assert_equal(act_len, len_index)

    def test_pad_arrays(self):
        # call internal functions
        split_idx_dist = self.atg._get_neighbors_pymatgen(self.atoms)
        pad_c_index, pad_n_index, pad_distances = self.atg._pad_arrays(
            self.atoms, *split_idx_dist
        )
        # check the shape to ensure padding
        act_shape = (len(self.atoms), self.atg.max_neigh)
        index_shape = pad_n_index.shape
        dist_shape = pad_distances.shape
        np.testing.assert_equal(act_shape, index_shape)
        np.testing.assert_equal(act_shape, dist_shape)

    def test_reshape_features(self):
        # call internal functions
        split_idx_dist = self.atg._get_neighbors_pymatgen(self.atoms)
        padded_idx_dist = self.atg._pad_arrays(self.atoms, *split_idx_dist)
        edge_index, all_distances = self.atg._reshape_features(
            *padded_idx_dist
        )
        # check the shapes of various tensors
        # combining c_index and n_index for edge_index
        # 2 arrays of length len(self.atoms) * self.atg.max_neigh
        act_edge_index_shape = (2, len(self.atoms) * self.atg.max_neigh)
        edge_index_shape = edge_index.size()
        np.testing.assert_equal(act_edge_index_shape, edge_index_shape)
        # check all_distances
        act_all_distances_shape = (len(self.atoms) * self.atg.max_neigh,)
        all_distances_shape = all_distances.size()
        np.testing.assert_equal(act_all_distances_shape, all_distances_shape)

    def test_convert(self):
        # run convert on a single atoms obj
        data = self.atg.convert(self.atoms)
        # check shape/values of features
        # edge index
        act_edge_index_shape = (2, len(self.atoms) * self.atg.max_neigh)
        edge_index_shape = data.edge_index.size()
        np.testing.assert_equal(act_edge_index_shape, edge_index_shape)
        # atomic numbers
        act_atomic_numbers = self.atoms.get_atomic_numbers()
        atomic_numbers = data.atomic_numbers.numpy()
        np.testing.assert_equal(act_atomic_numbers, atomic_numbers)
        # positions
        act_positions = self.atoms.get_positions()
        positions = data.pos.numpy()
        np.testing.assert_allclose(act_positions, positions)
        # check energy value
        act_energy = self.atoms.get_potential_energy(apply_constraint=False)
        test_energy = data.y
        np.testing.assert_equal(act_energy, test_energy)
        # forces
        act_forces = self.atoms.get_forces(apply_constraint=False)
        forces = data.force.numpy()
        np.testing.assert_allclose(act_forces, forces)
        # distances
        act_distances_shape = (len(self.atoms) * self.atg.max_neigh,)
        distances_shape = data.distances.size()
        np.testing.assert_equal(act_distances_shape, distances_shape)

    def test_convert_all(self):
        # run convert_all on a list with one atoms object
        # this does not test the atoms.db functionality
        atoms_list = [self.atoms]
        data_list = self.atg.convert_all(atoms_list)
        # check shape/values of features
        # edge index
        act_edge_index_shape = (2, len(self.atoms) * self.atg.max_neigh)
        edge_index_shape = data_list[0].edge_index.size()
        np.testing.assert_equal(act_edge_index_shape, edge_index_shape)
        # atomic numbers
        act_atomic_nubmers = self.atoms.get_atomic_numbers()
        atomic_numbers = data_list[0].atomic_numbers.numpy()
        np.testing.assert_equal(act_atomic_nubmers, atomic_numbers)
        # positions
        act_positions = self.atoms.get_positions()
        positions = data_list[0].pos.numpy()
        np.testing.assert_allclose(act_positions, positions)
        # check energy value
        act_energy = self.atoms.get_potential_energy(apply_constraint=False)
        test_energy = data_list[0].y
        np.testing.assert_equal(act_energy, test_energy)
        # forces
        act_forces = self.atoms.get_forces(apply_constraint=False)
        forces = data_list[0].force.numpy()
        np.testing.assert_allclose(act_forces, forces)
        # distances
        act_distances_shape = (len(self.atoms) * self.atg.max_neigh,)
        distances_shape = data_list[0].distances.size()
        np.testing.assert_equal(act_distances_shape, distances_shape)
