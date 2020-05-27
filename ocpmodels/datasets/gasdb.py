"""
Note that much of this code was taken from
`https://github.com/ulissigroup/cgcnn/`, which in turn was based on
`https://github.com/txie-93/cgcnn`.
"""

import os

import ase.db
import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data, DataLoader

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import collate
from ocpmodels.datasets import BaseDataset
from ocpmodels.datasets.elemental_embeddings import EMBEDDINGS

# Import the correct TQDM depending on where we are
try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


@registry.register_dataset("gasdb")
class Gasdb(BaseDataset):
    def __init__(self, config, transform=None, pre_transform=None):
        super(BaseDataset, self).__init__(config, transform, pre_transform)

        self.config = config
        ase_db_path = self.raw_file_names[0]
        self.ase_db = ase.db.connect(ase_db_path)

        try:
            self.data, self.slices = torch.load(self.processed_file_names[0])
            print(
                "### Loaded preprocessed data from:  {}".format(
                    self.processed_file_names
                )
            )
        except FileNotFoundError:
            self.process()

        assert self.ase_db.count() > 0

    @property
    def raw_file_names(self):
        r"""This method/property will use the first `*.db` object in the source
        directory."""
        for file_ in os.listdir(self.config["src"]):
            if file_.endswith(".db"):
                raw_file_name = os.path.join(self.config["src"], file_)
                return [raw_file_name]

    @property
    def processed_file_names(self):
        return [os.path.join(self.config["src"], "data.pt")]

    def process(self):
        print(
            "### Preprocessing atoms objects from:  {}".format(
                self.raw_file_names[0]
            )
        )
        feature_generator = AtomicFeatureGenerator(self.ase_db)
        try:
            energies = [row.adsorption_energy for row in self.ase_db.select()]
        except AttributeError:
            energies = [row.data['adsorption_energy'] for row in self.ase_db.select()]

        data_list = []
        zipped_data = zip(feature_generator, energies)
        for (embedding, distance, index), energy in tqdm(
            zipped_data,
            desc="preprocessing atomic features",
            total=len(energies),
            unit="structure",
        ):

            edge_index = [[], []]
            edge_attr = torch.FloatTensor(
                index.shape[0] * index.shape[1], distance.shape[-1]
            )
            for j in range(index.shape[0]):
                for k in range(index.shape[1]):
                    edge_index[0].append(j)
                    edge_index[1].append(index[j, k])
                    edge_attr[j * index.shape[1] + k] = distance[j, k].clone()
            edge_index = torch.LongTensor(edge_index)
            data_list.append(
                Data(
                    x=embedding,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=energy,
                    pos=None,
                )
            )

        self.data, self.slices = collate(data_list)
        torch.save((self.data, self.slices), self.processed_file_names[0])

    def get_full_dataloader(self, batch_size):
        data_loader = DataLoader(self, batch_size=batch_size)
        return data_loader


class AtomicFeatureGenerator:
    """
    Iterator meant to generate the features of the atoms objects within an `ase.db`

    Parameters
    ----------

    ase_db: instance of `ase.db.sqlite`
        SQL database of `ase.Atoms` objects
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance

    Returns
    -------

    embeddings: torch.Tensor shape (n_i, atom_fea_len)
    gaussian_distances: torch.Tensor shape (n_i, M, nbr_fea_len)
    all_indices: torch.LongTensor shape (n_i, M)
    """

    def __init__(
        self, ase_db, max_num_nbr=12, radius=6, dmin=0, step=0.2, start=0
    ):
        self.ase_db = ase_db
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.num = start

    def __iter__(self):
        return self

    def __len__(self):
        return self.ase_db.count()

    def __next__(self):
        try:
            item = self.__getitem__(self.num)
            self.num += 1
            return item

        except KeyError:
            assert self.num == self.__len__()
            raise StopIteration

    def __getitem__(self, index):
        atoms = self.ase_db.get_atoms(index + 1)
        return self.extract_atom_features(atoms)

    def extract_atom_features(self, atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)

        # `all_neighbors` is a nested list containing the neighbors of each
        # atom. The leaf objects are `structure.Neighbor` classes of pymatgen,
        # which are actually just 4-tuples containing the sites, distances,
        # indices, and images of each atoms' neighbors, respectively.
        all_neighbors = [
            sorted(
                neighbors, key=lambda neighbor: neighbor[1]
            )  # sorted by distance
            for neighbors in structure.get_all_neighbors(
                r=self.radius, include_index=True
            )
        ]

        # Grab the distances and indices of each atoms' neighbors
        all_distances, all_indices = [], []
        dummy_distance, dummy_index = self.radius + 1, 0
        for neighbors in all_neighbors:
            if len(neighbors) == 0:
                distances, indices = [dummy_distance], [dummy_index]
            else:
                # If the following throws an error, update to pymatgen>=2020.4.2
                _, distances, indices, _ = list(zip(*neighbors))
                distances = list(distances)
                indices = list(indices)
            # Pad empty elements in the features
            if len(distances) < self.max_num_nbr:
                padding_length = self.max_num_nbr - len(distances)
                distances.extend([dummy_distance] * padding_length)
                indices.extend([dummy_index] * padding_length)

            # Trim extra elements
            elif len(distances) > self.max_num_nbr:
                distances = distances[: self.max_num_nbr]
                indices = indices[: self.max_num_nbr]

            # Concatenation/formatting
            all_distances.append(distances)
            all_indices.append(indices)
        all_distances, all_indices = (
            np.array(all_distances),
            np.array(all_indices),
        )

        # Calculate the outputs
        gaussian_distances = self.gdf.expand(all_distances)
        embeddings = np.vstack(
            [EMBEDDINGS[site.specie.number] for site in structure]
        )

        # Turn everything into tensors
        embeddings = torch.Tensor(embeddings)
        gaussian_distances = torch.Tensor(gaussian_distances)
        all_indices = torch.LongTensor(all_indices)
        return embeddings, gaussian_distances, all_indices


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(
            -(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2
        )
