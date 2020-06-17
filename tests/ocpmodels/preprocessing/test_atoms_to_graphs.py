import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

import pytest
from new_atoms_to_graphs import AtomsToGraphs, expand_distance_gauss


@pytest.fixture(scope="class")
def atoms_to_graphs_internals(request):
    atoms = read("../test_atoms.json", index=0, format="json")
    test_object = AtomsToGraphs(
        max_neigh=12, radius=6, dummy_distance=7, dummy_index=-1
    )
    # test_object._get_neighbors_pymatgen(atoms)
    # test_object._pad_arrays()
    # test_object._gen_features()
    request.cls.atg = test_object
    request.cls.atoms = atoms


@pytest.mark.usefixtures("atoms_to_graphs_internals")
class TestAtomsToGraphs:
    def test_gen_neighbors_pymatgen(self):
        # call the internal function
        self.atg._get_neighbors_pymatgen(self.atoms)
        act_struct = AseAtomsAdaptor.get_structure(self.atoms)
        # use the old pymatgen method to get distance and indicies
        act_neigh = act_struct.get_all_neighbors(r=self.atg.radius)
        # randomly chose to check the neighbors and index of atom 4
        _, distances, indices, _ = zip(*act_neigh[4])
        act_dist = np.sort(distances)
        act_index = np.sort(indices)
        test_dist = np.sort(self.atg.split_n_distances[4])
        test_index = np.sort(self.atg.split_n_index[4])
        # check that the distance and neighbor index values are correct
        np.testing.assert_allclose(act_dist, test_dist)
        np.testing.assert_array_equal(act_index, test_index)
        # check for the correct length
        # the number of neighbors varies so hard to test that
        act_len = len(self.atoms)
        len_dist = len(self.atg.split_n_distances)
        len_index = len(self.atg.split_n_index)
        np.testing.assert_equal(act_len, len_dist)
        np.testing.assert_equal(act_len, len_index)

    def test_pad_arrays(self):
        # call internal function
        self.atg._pad_arrays()
        # check the shape to ensure padding
        act_shape = (len(self.atoms), self.atg.max_neigh)
        index_shape = self.atg.all_n_index.shape
        dist_shape = self.atg.all_distances.shape
        np.testing.assert_equal(act_shape, index_shape)
        np.testing.assert_equal(act_shape, dist_shape)

    def test_gen_features(self):
        # call internal function
        self.atg._gen_features()
        # check the shapes of various arrays or tensors
        # gaussian distances -> edge_attr
        len_gauss_expand = len(np.arange(0, 6 + 0.2, 0.2))
        act_gauss_dist_shape = (
            len(self.atoms) * self.atg.max_neigh,
            len_gauss_expand,
        )
        gauss_dist_shape = self.atg.gauss_distances.shape
        edge_attr_shape = self.atg.edge_attr.size()
        np.testing.assert_equal(act_gauss_dist_shape, gauss_dist_shape)
        np.testing.assert_equal(act_gauss_dist_shape, edge_attr_shape)
        # embeddings
        act_embed_shape = (len(self.atoms), 92)
        embed_shape = self.atg.embeddings.size()
        np.testing.assert_equal(act_embed_shape, embed_shape)
        # c_index, n_index -> edge_index
        act_n_index_shape = (len(self.atoms) * self.atg.max_neigh,)
        n_index_shape = self.atg.all_n_index.shape
        np.testing.assert_equal(act_n_index_shape, n_index_shape)
        # combining c_index and n_index for edge_index
        # 2 arrays of length len(self.atoms) * self.atg.max_neigh
        act_edge_index_shape = (2, len(self.atoms) * self.atg.max_neigh)
        edge_index_shape = self.atg.edge_index.size()
        np.testing.assert_equal(act_edge_index_shape, edge_index_shape)

    def test_convert(self):
        # clear self from previous tests
        self.atg._clear()
        # assert self.atg.atoms == None
        # run convert on single atoms
        # this does not test the atoms.db functionality
        atoms_list = [self.atoms]
        self.atg.convert(atoms_list)
        # check energy value
        act_energy = self.atoms.get_potential_energy()
        test_energy = self.atg.data_list[0].y
        np.testing.assert_equal(act_energy, test_energy)
        # check shape of other features
        # embeddings
        act_embed_shape = (len(self.atoms), 92)
        embed_shape = self.atg.data_list[0].x.size()
        np.testing.assert_equal(act_embed_shape, embed_shape)
        # edge index
        act_edge_index_shape = (2, len(self.atoms) * self.atg.max_neigh)
        edge_index_shape = self.atg.data_list[0].edge_index.size()
        np.testing.assert_equal(act_edge_index_shape, edge_index_shape)
        # edge attr
        len_gauss_expand = len(np.arange(0, 6 + 0.2, 0.2))
        act_edge_attr_shape = (
            len(self.atoms) * self.atg.max_neigh,
            len_gauss_expand,
        )
        edge_attr_shape = self.atg.data_list[0].edge_attr.size()
        np.testing.assert_equal(act_edge_attr_shape, edge_attr_shape)


def test_expand_distance_gauss():
    test_dists = np.array([3.0])
    test_gauss_dist = expand_distance_gauss(
        test_dists, dmin=0, dmax=6, step=0.5, var=None
    )
    gauss_range = np.arange(0, 6 + 0.5, 0.5)
    # check the shape returned
    act_gauss_dist_shape = (len(test_dists), len(gauss_range))
    gauss_dist_shape = test_gauss_dist.shape
    np.testing.assert_equal(act_gauss_dist_shape, gauss_dist_shape)
    # check value of index 1
    act_gauss_value = np.exp(-((3.0 - 0.5) ** 2) / 0.5 ** 2)
    gauss_value = test_gauss_dist[0][1]
    np.testing.assert_allclose(act_gauss_value, gauss_value)
