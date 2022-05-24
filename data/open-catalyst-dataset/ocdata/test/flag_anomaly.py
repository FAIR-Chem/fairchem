import math
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from ase.geometry import find_mic
from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from ase.io import read, write
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import SiteCollection
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.local_env import VoronoiNN

class DetectTrajAnomaly:
    def __init__(self, init_atoms, final_atoms, atoms_tag):
        """
        flag anomalies based on initial and final stucture of a relaxation.
        Args:
               init_atoms         `ase.Atoms` of the adslab in its initial state
               final_atoms        `ase.Atoms` of the adslab in its final state
               atoms_tag (list)    0=bulk, 1=surface, 2=adsorbate
        """
        self.init_atoms = init_atoms
        self.final_atoms = final_atoms
        self.atoms_tag = atoms_tag

    def is_adsorbate_dissociated(self):
        """
        True if the adsorbate is dissociated.
        """
        # test to see if adsorbate is dissociated from surface
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag==2]

        # if adsorbate is dissociated, the connectivity matrix would change
        initial_connectivity = self._get_connectivity(self.init_atoms[adsorbate_idx])
        final_connectivity = self._get_connectivity(self.final_atoms[adsorbate_idx])
        return np.array_equal(initial_connectivity, final_connectivity) is False

    def is_surface_reconstructed(self, slab_movement_thres=1):
        """
        if any slab atoms moved more than X Angstrom, consider possible reconstruction.
        A larger X means the user is more conversative of what's considered reconstructed.
        """
        slab_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag!=2]
        slab_init = self.init_atoms[slab_idx]
        slab_final = self.final_atoms[slab_idx]
        max_slab_movement = self._find_max_movement(slab_init, slab_final)
        return max_slab_movement >= slab_movement_thres

    def is_adsorbate_desorbed(self, neighbor_thres=3):
        """
        if the adsorbate binding atoms have no connection with slab atoms,
        consider it desorbed (returns True).
        Args:
            neighbor_thres    Given an atom, threshold (angstorm) for getting connecting
                              neighbor atoms, 3 is a fairly reasonable number.
        """
        adsorbate_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag==2]
        surface_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag==1]
        adslab_struct = AseAtomsAdaptor.get_structure(self.final_atoms)

        vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=6)
        for idx in adsorbate_atoms_idx:
            neighbors = adslab_struct.get_neighbors(adslab_struct[idx], neighbor_thres)
            surface_neighbors = [n.index for n in neighbors if n.index in surface_atoms_idx]
            if len(surface_neighbors) != 0:
                return False
        return True

    def _find_max_movement(self, init_atoms, final_atoms):
        '''
        Given ase.Atoms objects, find the furthest distance that any single atom in
        a set of atoms traveled (in Angstroms)
        Args:
                init_atoms      `ase.Atoms` of the structure in its initial state
                final_atoms     `ase.Atoms` of the structure in its final state
        Returns:
                max_movement    A float indicating the further movement of any single atom
                                before and after relaxation (in Angstroms)
        '''
        # Calculate the distances for each atom
        distances = final_atoms.positions - init_atoms.positions

        # Reduce the distances in case atoms wrapped around (the minimum image convention)
        _, movements = find_mic(distances, final_atoms.cell, final_atoms.pbc)
        max_movement = max(movements)
        return max_movement

    def _get_connectivity(self, atoms):
        """
        Generate the connectivity of an atoms obj.
        Args:
                    atoms      An `ase.Atoms` object
        Returns:
                    matrix     The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms)
        neighborList = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
        neighborList.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighborList.nl).toarray()
        return matrix
