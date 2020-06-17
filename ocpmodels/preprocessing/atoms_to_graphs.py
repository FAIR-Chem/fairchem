import ase.db
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from ocpmodels.common.utils import collate
from ocpmodels.datasets.elemental_embeddings import EMBEDDINGS
from torch_geometric.data import Data

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class AtomsToGraphs:
    def __init__(
        self, max_neigh=12, radius=6, dummy_distance=7, dummy_index=-1
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.dummy_distance = dummy_distance
        self.dummy_index = dummy_index
        self.data_list = []

    def _get_neighbors_pymatgen(self, atoms):
        self.atoms = atoms
        self.struct = AseAtomsAdaptor.get_structure(self.atoms)
        # these return jagged arrays meaning certain atoms have more neighbors than others
        _c_index, n_index, _images, n_distance = self.struct.get_neighbor_list(
            r=self.radius
        )
        # find the delimiters, the number of neighbors varies
        delim = np.where(np.diff(_c_index))[0] + 1
        # split the neighbor distance and neighbor index based on delimiter
        self.split_n_distances = np.split(n_distance, delim)
        self.split_n_index = np.split(n_index, delim)

    def _pad_arrays(self):
        # c_index is the center index, the atom to find the neighbors of
        c_index = np.arange(0, len(self.atoms), 1)
        # pad c_index
        self.all_c_index = np.repeat(c_index, self.max_neigh)
        # add dummy variables to desired array length
        self.all_n_index = np.full(
            (len(self.atoms), self.max_neigh), self.dummy_index
        )
        self.all_distances = np.full(
            (len(self.atoms), self.max_neigh), float(self.dummy_distance)
        )

        # loop over the stucture and replace dummy variables where values exist
        for i, (n_index, distances) in enumerate(
            zip(self.split_n_index, self.split_n_distances)
        ):
            if len(distances) == self.max_neigh:
                self.all_n_index[i] = n_index
                self.all_distances[i] = distances
                continue
            # padding arrays
            elif len(distances) < self.max_neigh:
                n_len = len(distances)
                # potentially add if n_len == 0: print (increase radius)
                self.all_n_index[i][:n_len] = n_index
                self.all_distances[i][:n_len] = distances
                continue
            # removing extra values so the length is equal to max_neigh
            # values are sorted by distance so only nearest neighbors are kept
            elif len(distances) > self.max_neigh:
                # this sorts the list min -> max and returns the indicies
                sorted_dist_i = np.argsort(distances)
                self.all_n_index[i] = n_index[sorted_dist_i[: self.max_neigh]]
                self.all_distances[i] = distances[
                    sorted_dist_i[: self.max_neigh]
                ]

    def _gen_features(self):
        # expand distances in gaussian basis
        gaussian_distances = expand_distance_gauss(self.all_distances)
        # reshape the array
        self.gauss_distances = gaussian_distances.reshape(
            (
                gaussian_distances.shape[0] * gaussian_distances.shape[1],
                gaussian_distances.shape[2],
            )
        )
        # convert to torch tensor, gauss_distances == edge_attr
        self.edge_attr = torch.FloatTensor(self.gauss_distances)
        # One-hot encoding for atom type embeddings
        embeddings = np.vstack(
            [EMBEDDINGS[site.specie.number] for site in self.struct]
        )
        # conver to torch tensor
        self.embeddings = torch.Tensor(embeddings)
        # edge_index
        self.all_n_index = self.all_n_index.reshape(
            (self.all_n_index.shape[0] * self.all_n_index.shape[1],)
        )
        self.edge_index = torch.LongTensor(
            [self.all_c_index, self.all_n_index]
        )

    def _clear(self):
        self.atoms = None
        self.struct = None
        self.embedding = None
        self.edge_index = None
        self.edge_attr = None

    def convert(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
    ):

        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            energy = atoms.get_potential_energy()
            self._get_neighbors_pymatgen(atoms)
            self._pad_arrays()
            self._gen_features()

            # put data in torch geometric format
            data = Data(
                x=self.embeddings,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=energy,
                pos=None,
            )
            self.data_list.append(data)
            # clears temp self variables
            self._clear()

        if collate_and_save:
            data, slices = collate(self.data_list)
            torch.save((data, slices), processed_file_path)


def expand_distance_gauss(distances, dmin=0, dmax=6, step=0.2, var=None):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    """
    Parameters
    ----------
    distance: np.array shape n-d array
      A distance matrix of any shape
    dmin: float
      Minimum interatomic distance
    dmax: float
      Maximum interatomic distance
    step: float
      Step size for the Gaussian filter

    Returns
    -------
    expanded_distance: shape (n+1)-d array
      Expanded distance matrix with the last dimension of length
      len(self.filter)
    """
    assert dmin < dmax
    assert dmax - dmin > step
    _filter = np.arange(dmin, dmax + step, step)
    if var is None:
        var = step

    return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var ** 2)
