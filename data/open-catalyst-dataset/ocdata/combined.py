
import catkit
import numpy as np
import warnings

from ase import neighborlist
from ase.neighborlist import natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN
from .constants import COVALENT_RADIUS
from .surfaces import constrain_surface


class Combined():
    '''
    This class handles all things with the adsorbate placed on a surface
    Needs one adsorbate and one surface to create this class.

    Attributes
    ----------
    adsorbate : Adsorbate
        object representing the adsorbate
    surface : Surface
        object representing the surface
    enumerate_all_configs : boolean
        whether to enumerate all adslab placements instead of choosing one random
    adsorbed_surface_atoms : list
        `Atoms` objects containing both the adsorbate and surface for all desired placements
    adsorbed_surface_sampling_strs : list
        list of strings capturing the config index for each adslab placement
    constrained_adsorbed_surfaces : list
        list of all constrained adslab atoms
    all_sites : list
        list of binding coordinates for all the adslab configs

    Public methods
    --------------
    get_adsorbed_bulk_dict(ind)
        returns a dict of info for the adsorbate+surface of the specified config index
    '''

    def __init__(self, adsorbate, surface, enumerate_all_configs):
        '''
        Adds adsorbate to surface, does the constraining, and aggregates all data necessary to write out.
        Can either pick a random configuration or store all possible ones.

        Args:
            adsorbate: the `Adsorbate` object
            surface: the `Surface` object
            enumerate_all_configs: whether to enumerate all adslab placements instead of choosing one random
        '''
        self.adsorbate = adsorbate
        self.surface = surface
        self.enumerate_all_configs = enumerate_all_configs

        self.add_adsorbate_onto_surface(self.adsorbate.atoms, self.surface.surface_atoms, self.adsorbate.bond_indices)

        self.constrained_adsorbed_surfaces = []
        self.all_sites = []
        for atoms in self.adsorbed_surface_atoms:
            # Add appropriate constraints
            self.constrained_adsorbed_surfaces.append(constrain_surface(atoms))

            # Do the hashing
            self.all_sites.append(self.find_sites(self.surface.constrained_surface, self.constrained_adsorbed_surfaces[-1], self.adsorbate.bond_indices))


    def add_adsorbate_onto_surface(self, adsorbate, surface, bond_indices):
        '''
        There are a lot of small details that need to be considered when adding an
        adsorbate onto a surface. This function will take care of those details for
        you.

        Args:
            adsorbate: An `ase.Atoms` object of the adsorbate
            surface: An `ase.Atoms` object of the surface
            bond_indices: A list of integers indicating the indices of the
                          binding atoms of the adsorbate
        Sets these values:
            adsorbed_surface_atoms: An `ase graphic Atoms` object containing the adsorbate and
                                    surface. The bulk atoms will be tagged with `0`; the
                                    surface atoms will be tagged with `1`, and the the
                                    adsorbate atoms will be tagged with `2` or above.
            adsorbed_surface_sampling_strs: String specifying the sample, [index]/[total]
                                            of reasonable adsorbed surfaces
        '''
        # convert surface atoms into graphic atoms object
        surface_gratoms = catkit.Gratoms(surface)
        surface_atom_indices = [i for i, atom in enumerate(surface)
                                if atom.tag == 1]
        surface_gratoms.set_surface_atoms(surface_atom_indices)
        surface_gratoms.pbc = np.array([True, True, False])

        # set up the adsorbate into graphic atoms object
        # with its connectivity matrix
        adsorbate_gratoms = self.convert_adsorbate_atoms_to_gratoms(adsorbate, bond_indices)

        # generate all possible adsorption configurations on that surface.
        # The "bonds" argument automatically take care of mono vs.
        # bidentate adsorption configuration.
        builder = catkit.gen.adsorption.Builder(surface_gratoms)
        with warnings.catch_warnings(): # suppress potential square root warnings
            warnings.simplefilter('ignore')
            adsorbed_surfaces = builder.add_adsorbate(adsorbate_gratoms,
                                                      bonds=bond_indices,
                                                      index=-1)

        # Filter out unreasonable structures.
        # Then pick one from the reasonable configurations list as an output.
        reasonable_adsorbed_surfaces = [surface for surface in adsorbed_surfaces
                                        if self.is_config_reasonable(surface)]

        self.adsorbed_surface_atoms = []
        self.adsorbed_surface_sampling_strs = []
        if self.enumerate_all_configs:
            self.num_configs = len(reasonable_adsorbed_surfaces)
            for ind, reasonable_config in enumerate(reasonable_adsorbed_surfaces):
                self.adsorbed_surface_atoms.append(reasonable_config)
                self.adsorbed_surface_sampling_strs.append(str(ind) + '/' + str(len(reasonable_adsorbed_surfaces)))
        else:
            self.num_configs = 1
            reasonable_adsorbed_surface_index = np.random.choice(len(reasonable_adsorbed_surfaces))
            self.adsorbed_surface_atoms.append(reasonable_adsorbed_surfaces[reasonable_adsorbed_surface_index])
            self.adsorbed_surface_sampling_strs.append(str(reasonable_adsorbed_surface_index) + '/' + str(len(reasonable_adsorbed_surfaces)))

    def convert_adsorbate_atoms_to_gratoms(self, adsorbate, bond_indices):
        """
        Convert adsorbate atoms object into graphic atoms object,
        so the adsorbate can be placed onto the surface with optimal
        configuration. Set tags for adsorbate atoms to 2, to distinguish
        them from surface atoms.

        Args:
            adsorbate           An `ase.Atoms` object of the adsorbate
            bond_indices          A list of integers indicating the indices of the
                                  binding atoms of the adsorbate

        Returns:
            adsorbate_gratoms   An graphic atoms object of the adsorbate.
        """
        connectivity = self.get_connectivity(adsorbate)
        adsorbate_gratoms = catkit.Gratoms(adsorbate, edges=connectivity)
        # tag adsorbate atoms: non-binding atoms as 2, the binding atom(s) as 3 for now to
        # track adsorption site for analyzing if adslab configuration is reasonable.
        adsorbate_gratoms.set_tags([3 if idx in bond_indices else 2 for idx in range(len(adsorbate_gratoms))])
        return adsorbate_gratoms

    def get_connectivity(self, adsorbate):
        """
        Generate the connectivity of an adsorbate atoms obj.

        Args:
            adsorbate  An `ase.Atoms` object of the adsorbate

        Returns:
            matrix     The connectivity matrix of the adsorbate.
        """
        cutoff = natural_cutoffs(adsorbate)
        neighborList = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
        neighborList.update(adsorbate)
        matrix = neighborlist.get_connectivity_matrix(neighborList.nl).toarray()
        return matrix

    def is_config_reasonable(self, adslab):
        """
        Function that check whether the adsorbate placement is reasonable.
        Two criteria are: 1. The adsorbate should be placed on the slab:
        the fractional coordinates of the adsorption site is bounded by the unit cell.
        2. The adsorbate should not be buried into the surface: for any atom
        in the adsorbate, if the distance between the atom and slab atoms
        are closer than 80% of their expected covalent bond, we reject that placement.

        Args:
            adslab          An `ase.Atoms` object of the adsorbate+slab complex.

        Returns:
            A boolean indicating whether or not the adsorbate placement is
            reasonable.
        """
        vnn = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)
        adsorbate_indices = [atom.index for atom in adslab if atom.tag >= 2]
        adsorbate_bond_indices = [atom.index for atom in adslab if atom.tag == 3]
        structure = AseAtomsAdaptor.get_structure(adslab)
        slab_lattice = structure.lattice

        # Check to see if the fractional coordinates of the adsorption site is bounded
        # by the slab unit cell. We loosen the threshold to -0.01 and 1.01
        # to not wrongly exclude reasonable edge adsorption site.
        for idx in adsorbate_bond_indices:
            coord = slab_lattice.get_fractional_coords(structure[idx].coords)
            if np.any((coord < -0.01) | (coord > 1.01)):
                return False

        # Then, check the covalent radius between each adsorbate atoms
        # and its nearest neighbors that are slab atoms
        # to make sure adsorbate is not buried into the surface
        for idx in adsorbate_indices:
            try:
                nearneighbors = vnn.get_nn_info(structure, n=idx)
            except ValueError:
                return False

            slab_nn = [nn for nn in nearneighbors if nn['site_index'] not in adsorbate_indices]
            for nn in slab_nn:
                ads_elem = structure[idx].species_string
                nn_elem = structure[nn['site_index']].species_string
                cov_bond_thres = 0.8 * (COVALENT_RADIUS[ads_elem] + COVALENT_RADIUS[nn_elem])/100
                actual_dist = adslab.get_distance(idx, nn['site_index'], mic=True)
                if actual_dist < cov_bond_thres:
                    return False

        # If the structure is reasonable, change tags of adsorbate atoms from 2 and 3 to 2 only
        # for ML model compatibility and data cleanliness of the output adslab configurations
        old_tags = adslab.get_tags()
        adslab.set_tags(np.where(old_tags == 3, 2, old_tags))
        return True

    def find_sites(self, surface, adsorbed_surface, bond_indices):
        '''
        Finds the Cartesian coordinates of the bonding atoms of the adsorbate.

        Args:
            surface             `ase.Atoms` of the chosen surface
            adsorbed_surface    An `ase graphic Atoms` object containing the
                                adsorbate and surface.
            bond_indices        A list of integers indicating the indices of the
                                binding atoms of the adsorbate
        Returns:
            sites   A tuple of 3-tuples containing the Cartesian coordinates of
                    each of the binding atoms
        '''
        sites = []
        for idx in bond_indices:
            binding_atom_index = len(surface) + idx
            atom = adsorbed_surface[binding_atom_index]
            positions = tuple(round(coord, 2) for coord in atom.position)
            sites.append(positions)

        return tuple(sites)

    def get_adsorbed_bulk_dict(self, ind):
        '''
        Returns an organized dict for writing to files.
        All info is already processed and stored in class variables.
        '''
        ads_sampling_str = self.adsorbate.adsorbate_sampling_str + "_" + self.adsorbed_surface_sampling_strs[ind]

        return {"adsorbed_bulk_atomsobject" : self.constrained_adsorbed_surfaces[ind],
                "adsorbed_bulk_metadata"    : (self.surface.bulk_object.mpid,
                                               self.surface.millers,
                                               round(self.surface.shift, 3),
                                               self.surface.top,
                                               self.adsorbate.smiles,
                                               self.all_sites[ind]),
                "adsorbed_bulk_samplingstr" : self.surface.overall_sampling_str + "_" + ads_sampling_str}
