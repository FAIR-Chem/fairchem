import numpy as np

from ase import neighborlist
from ase.neighborlist import natural_cutoffs


class DetectTrajAnomaly:
    def __init__(
        self,
        init_atoms,
        final_atoms,
        atoms_tag,
        final_slab_atoms=None,
        reconstruction_cutoff_multiplier=1.5,
    ):
        """
        Flag anomalies based on initial and final stucture of a relaxation.

        Args:
               init_atoms (ase.Atoms): the adslab in its initial state
               final_atoms (ase.Atoms): the adslab in its final state
               atoms_tag (list): the atom tags; 0=bulk, 1=surface, 2=adsorbate
               final_slab_atoms (ase.Atoms, optional): the relaxed slab if unspecified this defaults
                   to using the initial adslab instead.
               reconstruction_cutoff_multiplier (float): cushion for small atom movements when assessing
                    atom connectivity for reconstruction
        """
        self.init_atoms = init_atoms
        self.final_atoms = final_atoms
        self.final_slab_atoms = final_slab_atoms
        self.atoms_tag = atoms_tag
        self.initial_connectivity = self._get_connectivity(self.init_atoms)
        self.final_connectivity = self._get_connectivity(self.final_atoms)
        self.final_connectivity_desorption = self._get_connectivity(
            self.final_atoms, 1.3
        )
        if self.final_slab_atoms is not None:
            self.slab_connectivity = self._get_connectivity(
                self.final_slab_atoms, reconstruction_cutoff_multiplier
            )
        else:
            self.slab_connectivity = self._get_connectivity(
                self.final_atoms, reconstruction_cutoff_multiplier
            )

    def is_adsorbate_dissociated(self):
        """
        Tests if the initial adsorbate connectivity is maintained.

        Returns:
            (bool): True if the connectivity was not maintained, otherwise False
        """
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 2]
        return (
            np.array_equal(
                self.initial_connectivity[adsorbate_idx][:, adsorbate_idx],
                self.final_connectivity[adsorbate_idx][:, adsorbate_idx],
            )
            is False
        )

    def is_surface_reconstructed(self):
        """
        If any slab atoms moved more than X Angstrom, consider possible reconstruction.
        A larger X means the user is more conversative of what's considered reconstructed.

        Returns:
            (bool): True if the surface is reconstructed, otherwise False
        """
        surf_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
        final_slab_connectivity_from_adslab = self.final_connectivity[surf_idx][
            :, surf_idx
        ]

        return 1 in final_slab_connectivity_from_adslab - self.slab_connectivity

    def is_adsorbate_desorbed(self):
        """
        If the adsorbate binding atoms have no connection with slab atoms,
        consider it desorbed.

        Returns:
            (bool): True if there is desorption, otherwise False
        """
        adsorbate_atoms_idx = [
            idx for idx, tag in enumerate(self.atoms_tag) if tag == 2
        ]
        surface_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]

        for idx in adsorbate_atoms_idx:
            if sum(self.final_connectivity_desorption[idx][surface_atoms_idx]) >= 1:
                return False
        return True

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Generate the connectivity of an atoms obj.

        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity

        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = list(np.array(natural_cutoffs(atoms)) * cutoff_multiplier)
        neighborList = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        neighborList.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighborList.nl).toarray()
        return matrix
