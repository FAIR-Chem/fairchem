import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data

from ocpmodels.common.utils import collate

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Additionally, arrays are padded because not all atoms
    have the same number of neighbors. Lastly, atomic properties and the graph information are put into a
    PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        dummy_distance (int or float): A dummy distance to pad with, should be larger than radius cutoff.
        dummy_index (int): A dummy index to pad with.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        dummy_distance (int or float): A dummy distance to pad with.
        dummy_index (int): A dummy index to pad with.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
    """

    def __init__(
        self,
        max_neigh=12,
        radius=6,
        dummy_distance=7,
        dummy_index=-1,
        r_energy=False,
        r_forces=False,
        r_distances=False,
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.dummy_distance = dummy_distance
        self.dummy_index = dummy_index
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns split neighbors indices and distances"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        # these return jagged arrays meaning certain atoms have more neighbors than others
        _c_index, n_index, _images, n_distance = struct.get_neighbor_list(
            r=self.radius
        )
        # find the delimiters, the number of neighbors varies
        delim = np.where(np.diff(_c_index))[0] + 1
        # split the neighbor index, distance, lattice offsets based on delimiter
        split_n_index = np.split(n_index, delim)
        split_n_distances = np.split(n_distance, delim)
        split_offsets = np.split(_images, delim)

        return split_n_index, split_n_distances, split_offsets

    def _pad_arrays(
        self, atoms, split_n_index, split_n_distances, split_offsets
    ):
        """Pads arrays to standardize the length"""
        # c_index is the center index, the atom to find the neighbors of
        c_index = np.arange(0, len(atoms), 1)
        # pad c_index
        pad_c_index = np.repeat(c_index, self.max_neigh)
        # add dummy variables to desired array length
        pad_n_index = np.full((len(atoms), self.max_neigh), self.dummy_index)
        pad_distances = np.full(
            (len(atoms), self.max_neigh), float(self.dummy_distance)
        )
        pad_offsets = np.zeros((len(atoms), self.max_neigh, 3), dtype=np.int)

        # loop over the stucture and replace dummy variables where values exist
        for i, (n_index, distances, offsets) in enumerate(
            zip(split_n_index, split_n_distances, split_offsets)
        ):
            if len(distances) == self.max_neigh:
                pad_n_index[i] = n_index
                pad_distances[i] = distances
                pad_offsets[i] = offsets
                continue
            # padding arrays
            elif len(distances) < self.max_neigh:
                n_len = len(distances)
                # potentially add if n_len == 0: print (increase radius)
                pad_n_index[i][:n_len] = n_index
                pad_distances[i][:n_len] = distances
                pad_offsets[i][:n_len] = offsets
                continue
            # removing extra values so the length is equal to max_neigh
            # values are sorted by distance so only nearest neighbors are kept
            elif len(distances) > self.max_neigh:
                # this sorts the list min -> max and returns the indices
                sorted_dist_i = np.argsort(distances)
                pad_n_index[i] = n_index[sorted_dist_i[: self.max_neigh]]
                pad_distances[i] = distances[sorted_dist_i[: self.max_neigh]]
                pad_offsets[i] = offsets[sorted_dist_i[: self.max_neigh]]
        return pad_c_index, pad_n_index, pad_distances, pad_offsets

    def _reshape_features(
        self, pad_c_index, pad_n_index, pad_distances, pad_offsets
    ):
        """Processes the center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        # edge_index reshape, combine, and define in torch
        pad_n_index = pad_n_index.reshape(
            (pad_n_index.shape[0] * pad_n_index.shape[1],)
        )
        edge_index = torch.LongTensor([pad_c_index, pad_n_index])
        # reshape distance to match indices
        pad_distances = pad_distances.reshape(
            (pad_distances.shape[0] * pad_distances.shape[1],)
        )
        all_distances = torch.Tensor(pad_distances)
        # reshape offsets to match indices
        pad_offsets = pad_offsets.reshape(
            (pad_offsets.shape[0] * pad_offsets.shape[1], -1)
        )
        cell_offsets = torch.LongTensor(pad_offsets)
        return edge_index, all_distances, cell_offsets

    def convert(
        self, atoms,
    ):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # run internal functions to get padded indices and distances
        split_idx_dist = self._get_neighbors_pymatgen(atoms)
        padded_idx_dist = self._pad_arrays(atoms, *split_idx_dist)
        edge_index, all_distances, cell_offsets = self._reshape_features(
            *padded_idx_dist
        )

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)

        # put the minimum data in torch geometric data object
        data = Data(
            edge_index=edge_index,
            cell=cell,
            cell_offsets=cell_offsets,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=positions.shape[0],
        )

        # optionally include other properties
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces
        if self.r_distances:
            data.distances = all_distances

        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list
