from __future__ import annotations

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
        surface_change_cutoff_multiplier=1.5,
        desorption_cutoff_multiplier=1.5,
    ):
        """
        Flag anomalies based on initial and final stucture of a relaxation.

        Args:
            init_atoms (ase.Atoms): the adslab in its initial state
            final_atoms (ase.Atoms): the adslab in its final state
            atoms_tag (list): the atom tags; 0=bulk, 1=surface, 2=adsorbate
            final_slab_atoms (ase.Atoms, optional): the relaxed slab if unspecified this defaults
            to using the initial adslab instead.
            surface_change_cutoff_multiplier (float, optional): cushion for small atom movements
                when assessing atom connectivity for reconstruction
            desorption_cutoff_multiplier (float, optional): cushion for physisorbed systems to not
                be discarded. Applied to the covalent radii.
        """
        self.init_atoms = init_atoms
        self.final_atoms = final_atoms
        self.final_slab_atoms = final_slab_atoms
        self.atoms_tag = atoms_tag
        self.surface_change_cutoff_multiplier = surface_change_cutoff_multiplier
        self.desorption_cutoff_multiplier = desorption_cutoff_multiplier

        if self.final_slab_atoms is None:
            slab_idxs = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
            self.final_slab_atoms = self.init_atoms[slab_idxs]

    def is_adsorbate_dissociated(self):
        """
        Tests if the initial adsorbate connectivity is maintained.

        Returns:
            (bool): True if the connectivity was not maintained, otherwise False
        """
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 2]
        return not (
            np.array_equal(
                self._get_connectivity(self.init_atoms[adsorbate_idx]),
                self._get_connectivity(self.final_atoms[adsorbate_idx]),
            )
        )

    def has_surface_changed(self):
        """
        Tests bond breaking / forming events within a tolerance on the surface so
        that systems with significant adsorbate induces surface changes may be discarded
        since the reference to the relaxed slab may no longer be valid.

        Returns:
            (bool): True if the surface is reconstructed, otherwise False
        """
        surf_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]

        adslab_connectivity = self._get_connectivity(self.final_atoms[surf_idx])
        slab_connectivity_w_cushion = self._get_connectivity(
            self.final_slab_atoms, self.surface_change_cutoff_multiplier
        )
        slab_test = 1 in adslab_connectivity - slab_connectivity_w_cushion

        adslab_connectivity_w_cushion = self._get_connectivity(
            self.final_atoms[surf_idx], self.surface_change_cutoff_multiplier
        )
        slab_connectivity = self._get_connectivity(self.final_slab_atoms)
        adslab_test = 1 in slab_connectivity - adslab_connectivity_w_cushion

        return any([slab_test, adslab_test])

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
        final_connectivity = self._get_connectivity(
            self.final_atoms, self.desorption_cutoff_multiplier
        )

        for idx in adsorbate_atoms_idx:
            if sum(final_connectivity[idx][surface_atoms_idx]) >= 1:
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
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        ase_neighbor_list = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        ase_neighbor_list.update(atoms)
        return neighborlist.get_connectivity_matrix(ase_neighbor_list.nl).toarray()

    def is_adsorbate_intercalated(self):
        """
        Ensure the adsorbate isn't interacting with an atom that is not allowed to relax.

        Returns:
            (bool): True if any adsorbate atom neighbors a frozen atom, otherwise False
        """
        adsorbate_atoms_idx = [
            idx for idx, tag in enumerate(self.atoms_tag) if tag == 2
        ]
        frozen_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 0]
        final_connectivity = self._get_connectivity(
            self.final_atoms,
        )

        for idx in adsorbate_atoms_idx:
            if sum(final_connectivity[idx][frozen_atoms_idx]) >= 1:
                return True
        return False
