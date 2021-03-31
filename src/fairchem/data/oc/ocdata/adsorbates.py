
import numpy as np
import pickle


class Adsorbate():
    '''
    This class handles all things with the adsorbate.
    Selects one (either specified or random), and stores info as an object

    Attributes
    ----------
    atoms : Atoms
        actual atoms of the adsorbate
    smiles : str
        SMILES representation of the adsorbate
    bond_indices : list
        indices of the atoms meant to be bonded to the surface
    adsorbate_sampling_str : str
        string capturing the adsorbate index and total possible adsorbates
    '''

    def __init__(self, adsorbate_database, specified_index=None):
        self.choose_adsorbate_pkl(adsorbate_database, specified_index)

    def choose_adsorbate_pkl(self, adsorbate_database, specified_index=None):
        '''
        Chooses an adsorbate from our pkl based inverted index at random.

        Args:
            adsorbate_database: A string pointing to the a pkl file that contains
                                an inverted index over different adsorbates.
            specified_index: adsorbate index to choose instead of choosing a random one
        Sets:
            atoms                    `ase.Atoms` object of the adsorbate
            smiles                   SMILES-formatted representation of the adsorbate
            bond_indices             list of integers indicating the indices of the atoms in
                                     the adsorbate that are meant to be bonded to the surface
            adsorbate_sampling_str   Enum string specifying the sample, [index]/[total]
        '''
        with open(adsorbate_database, 'rb') as f:
            inv_index = pickle.load(f)

        if specified_index is not None:
            element = specified_index
        else:
            element = np.random.choice(len(inv_index))

        self.adsorbate_sampling_str = str(element) + "/" + str(len(inv_index))
        self.atoms, self.smiles, self.bond_indices = inv_index[element]
