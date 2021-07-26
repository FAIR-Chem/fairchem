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

def get_connectivity(atoms, multiple=1):
    """
    Generate the connectivity of an atoms obj.
    Args:
                atoms      An `ase.Atoms` object
    Returns:
                matrix     The connectivity matrix of the atoms object.
    """
    cutoff = natural_cutoffs(atoms, mult=multiple)
    neighborList = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    matrix = neighborlist.get_connectivity_matrix(neighborList.nl).toarray()
    return matrix

def find_max_movement(init_atoms, final_atoms):
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

def is_adsorbate_placed_correct(adslab, atoms_tag):
    """
    Make sure all adsorbate atoms are connected after placement.
    False means there is at least one isolated adsorbate atom.
    Args:
            adslab             `ase.Atoms` of the structure in its initial state
            atoms_tag (list)    0=bulk, 1=surface, 2=adsorbate
    Returns:
            boolean    If there is any stand alone adsorbate atoms after placement,
                       return False.

    """
    adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag==2]
    connectivity = get_connectivity(self.init_atoms[adsorbate_idx])
    if np.any(np.sum(connectivity, axis=0)==0):
        return False
    return True

class DetectAnomaly:
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

    def _is_adsorbate_dissociated(self):
        """
        True if the adsorbate is dissociated
        """
        # test to see if adsorbate is dissociated from surface
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag==2]

        # if adsorbate is dissociated, the connectivity matrix would change
        initial_connectivity = get_connectivity(self.init_atoms[adsorbate_idx])
        final_connectivity = get_connectivity(self.final_atoms[adsorbate_idx])
        if np.array_equal(initial_connectivity, final_connectivity):
            return False
        return True

    def _is_surface_reconstructed(self, slab_movement_thres):
        """
        if any slab atoms moved more than X Angstrom, consider possible reconstruction.
        A larger X means user is more conversative of what's considered reconstructed.
        """
        slab_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag!=2]
        slab_init = self.init_atoms[slab_idx]
        slab_final = self.final_atoms[slab_idx]
        max_slab_movement = find_max_movement(slab_init, slab_final)
        if max_slab_movement >= slab_movement_thres: # A
            return True
        return False

    def _is_adsorbate_desorbed(self, neighbor_thres=3):
        """
        if the adsorbate binding atoms have no connection with slab atoms,
        consider it desorbed.
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