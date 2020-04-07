'''
This submodule contains the scripts that the we used to sample the adsorption
structures.

Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
'''

__authors__ = ['Kevin Tran, Aini Palizhati']
__emails__ = ['ktran@andrew.cmu.edu', 'apalizha@andrew.cmu.edu']

import math
import random
import numpy as np
import catkit
import ase
import ase.db
from ase import neighborlist
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN


ELEMENTS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29:
            'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
            36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42:
            'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
            49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55:
            'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
            62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68:
            'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81:
            'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
            88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94:
            'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
            101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg',
            107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
            113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

# Covalent radius of elements (unit is pm, 1pm=0.01 angstrom)
# Value are taken from https://github.com/lmmentel/mendeleev
covalent_radius = {'H': 32.0, 'He': 46.0, 'O': 63.0, 'F': 64.0, 'Ne': 67.0, 'N': 71.0,
                   'C': 75.0, 'B': 85.0, 'Ar': 96.0, 'Cl': 99.0, 'Be': 102.0, 'S': 103.0,
                   'Ni': 110.0, 'P': 111.0, 'Co': 111.0, 'Cu': 112.0, 'Br': 114.0,
                   'Si': 116.0, 'Fe': 116.0, 'Se': 116.0, 'Kr': 117.0, 'Zn': 118.0,
                   'Mn': 119.0, 'Pd': 120.0, 'Ge': 121.0, 'As': 121.0, 'Rg': 121.0,
                   'Cr': 122.0, 'Ir': 122.0, 'Cn': 122.0, 'Pt': 123.0, 'Ga': 124.0,
                   'Au': 124.0, 'Ru': 125.0, 'Rh': 125.0, 'Al': 126.0, 'Tc': 128.0,
                   'Ag': 128.0, 'Ds': 128.0, 'Os': 129.0, 'Mt': 129.0, 'Xe': 131.0, 'Re': 131.0,
                   'Li': 133.0, 'I': 133.0, 'Hg': 133.0, 'V': 134.0, 'Hs': 134.0, 'Ti': 136.0,
                   'Cd': 136.0, 'Te': 136.0, 'Nh': 136.0, 'W': 137.0, 'Mo': 138.0, 'Mg': 139.0,
                   'Sn': 140.0, 'Sb': 140.0, 'Bh': 141.0, 'In': 142.0, 'Rn': 142.0, 'Sg': 143.0,
                   'Fl': 143.0, 'Tl': 144.0, 'Pb': 144.0, 'Po': 145.0, 'Ta': 146.0, 'Nb': 147.0,
                   'At': 147.0, 'Sc': 148.0, 'Db': 149.0, 'Bi': 151.0, 'Hf': 152.0, 'Zr': 154.0,
                   'Na': 155.0, 'Rf': 157.0, 'Og': 157.0, 'Lr': 161.0, 'Lu': 162.0, 'Mc': 162.0,
                   'Y': 163.0, 'Ce': 163.0, 'Tm': 164.0, 'Er': 165.0, 'Es': 165.0, 'Ts': 165.0,
                   'Ho': 166.0, 'Am': 166.0, 'Cm': 166.0, 'Dy': 167.0, 'Fm': 167.0, 'Eu': 168.0,
                   'Tb': 168.0, 'Bk': 168.0, 'Cf': 168.0, 'Gd': 169.0, 'Pa': 169.0, 'Yb': 170.0,
                   'U': 170.0, 'Ca': 171.0, 'Np': 171.0, 'Sm': 172.0, 'Pu': 172.0, 'Pm': 173.0,
                   'Md': 173.0, 'Nd': 174.0, 'Th': 175.0, 'Lv': 175.0, 'Pr': 176.0, 'No': 176.0,
                   'La': 180.0, 'Sr': 185.0, 'Ac': 186.0, 'K': 196.0, 'Ba': 196.0, 'Ra': 201.0,
                   'Rb': 210.0, 'Fr': 223.0, 'Cs': 232.0}

# We will enumerate surfaces with Miller indices <= MAX_MILLER
MAX_MILLER = 2

# We will create surfaces that are at least MIN_XY Angstroms wide. GASpy uses
# 4.5, but our larger adsorbates here can be up to 3.6 Angstroms long. So 4.5 +
# 3.6 ~= 8 Angstroms
MIN_XY = 8.


def sample_structures(bulk_database='bulks.db',
                      adsorbate_database='adsorbates.db',
                      n_species_weights=None):
    '''
    This parent function will randomly select an adsorption structure from a
    given set of bulks.

    Args:
        bulk_database       A string pointing to the ASE *.db object that
                            contains the bulks you want to consider.
        n_species_weights   A dictionary whose keys are integers containing the
                            number of species you want to consider and whose
                            values are the probabilities of selecting this
                            number. The probabilities must sum to 1.
    Returns:
        adsorbed_surface    `ase.Atoms` object containing a surface with a
                            random adsorbate placed on it. The bulk atoms
                            will be tagged with `0`; the surface atoms will be
                            tagged with `1`, and the the adsorbate atoms will
                            be tagged with `2` or above.
        surface             `ase.Atoms` object containing only the surface that
                            we sampled.
    '''
    # Choose which surface we want
    n_species = choose_n_species(n_species_weights)
    elements = choose_elements(bulk_database, n_species)
    bulk = choose_bulk(bulk_database, elements)
    surface = choose_surface(bulk)

    # Choose the adsorbate and place it on the surface
    adsorbate = choose_adsorbate(adsorbate_database)
    adsorbed_surface = add_adsorbate_onto_surface(surface, adsorbate)

    # Add appropriate constraints
    adsorbed_surface = constrain_surface(adsorbed_surface)
    surface = constrain_surface(surface)
    return adsorbed_surface, surface


def choose_n_species(n_species_weights):
    '''
    Chooses the number of species we should look for in this sample.

    Arg:
        n_species_weights   A dictionary whose keys are integers containing the
                            number of species you want to consider and whose
                            values are the probabilities of selecting this
                            number. The probabilities must sum to 1.
    Returns:
        n_species   An integer showing how many species have been chosen.
    '''
    if n_species_weights is None:
        n_species_weights = {1: 0.05, 2: 0.65, 3: 0.3}

    n_species = list(n_species_weights.keys())
    weights = list(n_species_weights.values())
    assert math.isclose(sum(weights), 1)

    n_species = np.random.choice(n_species, p=weights)
    return n_species


def choose_elements(bulk_database, n):
    '''
    Chooses `n` elements at random from the set of elements inside the given
    database.

    Args:
        bulk_database   A string pointing to the ASE *.db object that contains
                        the bulks you want to consider.
        n               A positive integer indicating how many elements you
                        want to choose.
    Returns:
        elements    A list of strings indicating the chosen elements
    '''
    db = ase.db.connect(bulk_database)
    candidate_combinations = {tuple(sorted(set(ELEMENTS[number] for number in row.numbers)))
                              for row in db.select() if len(set(row.numbers)) == n}
    try:
        elements = list(random.choice(list(candidate_combinations)))
        return elements

    except IndexError:
        raise ValueError('Randomly chose to look for a %i-component material, '
                         'but no such materials exist in %s. Please add one '
                         'to the database or change the weights to exclude '
                         'this number of components.' % (n, n, bulk_database))


def choose_bulk(bulk_database, elements):
    '''
    Chooses a bulks from our database at random as long as the bulk contains
    all the specified elements.

    Args:
        bulk_database   A string pointing to the ASE *.db object that contains
                        the bulks you want to consider.
        elements        A list of strings indicating the elements you want to
                        show up in the bulk. The strings much match one of the
                        values in the `ELEMENTS` constant in this submodule.
    Returns:
        atoms   `ase.Atoms` of the chosen bulk structure.
    '''
    db = ase.db.connect(bulk_database)
    all_atoms = [row.toatoms() for row in db.select(elements)]
    atoms = random.choice(all_atoms)
    return atoms


def choose_surface(bulk_atoms):
    '''
    Enumerates and chooses a random surface from a bulk structure.

    Arg:
        bulk_atoms          `ase.Atoms` object of the bulk you want to choose a
                             surfaces from.
    Returns:
        surface_atoms       `ase.Atoms` of the chosen surface
    '''
    surfaces = enumerate_surfaces(bulk_atoms)
    surface_struct = random.choice(surfaces)
    unit_surface_atoms = AseAtomsAdaptor.get_atoms(surface_struct)
    surface_atoms = tile_atoms(unit_surface_atoms)
    tag_surface_atoms(bulk_atoms, surface_atoms)
    return surface_atoms


def choose_adsorbate(adsorbate_database):
    '''
    Chooses a bulks from our database at random as long as the bulk contains
    all the specified elements.

    Args:
        adsorbate_database   A string pointing to the ASE *.db object that contains
                             the adsorbates you want to consider.
    Returns:
        adsorbate            A dictionary that has the following format:
                             {'atoms_obj': Atoms(...), 'bond_index': ...}
    '''
    db = ase.db.connect(adsorbate_database)
    all_adsorbates = [{'atoms_obj': row.toatoms(), 'bond_idx': row.data['bond_idx']} for row in db.select()]
    adsorbate = random.choice(all_adsorbates)
    return adsorbate


def enumerate_surfaces(bulk_atoms, max_miller=MAX_MILLER):
    '''
    Enumerate all the symmetrically distinct surfaces of a bulk structure. It
    will not enumerate surfaces with Miller indices above the `max_miller`
    argument. Note that we also look at the bottoms of surfaces if they are
    distinct from the top. If they are distinct, we flip the surface so the bottom
    is pointing upwards.

    Args:
        bulk_atoms  `ase.Atoms` object of the bulk you want to enumerate
                    surfaces from.
        max_miller  An integer indicating the maximum Miller index of the surfaces
                    you are willing to enumerate. Increasing this argument will
                    increase the number of surfaces, but the surfaces will
                    generally become larger.
    Returns:
        all_slabs   A list of `pymatgen.Structure` objects for each of the
                    surfaces we have enumerated.
    '''
    bulk_struct = standardize_bulk(bulk_atoms)

    all_slabs = []
    for miller_indices in get_symmetrically_distinct_miller_indices(bulk_struct, MAX_MILLER):
        slab_gen = SlabGenerator(initial_structure=bulk_struct,
                                 miller_index=miller_indices,
                                 min_slab_size=7.,
                                 min_vacuum_size=20.,
                                 lll_reduce=False,
                                 center_slab=True,
                                 primitive=True,
                                 max_normal_search=1)
        slabs = slab_gen.get_slabs(tol=0.3,
                                   bonds=None,
                                   max_broken_bonds=0,
                                   symmetrize=False)

        # If the bottoms of the slabs are different than the tops, then we want
        # to consider them, too
        flipped_slabs = [flip_struct(slab) for slab in slabs
                         if is_structure_invertible(slab) is False]
        slabs.extend(flipped_slabs)

        all_slabs.extend(slabs)
    return all_slabs


def standardize_bulk(atoms):
    '''
    There are many ways to define a bulk unit cell. If you change the unit cell
    itself but also change the locations of the atoms within the unit cell, you
    can get effectively the same bulk structure. To address this, there is a
    standardization method used to reduce the degrees of freedom such that each
    unit cell only has one "true" configuration. This function will align a
    unit cell you give it to fit within this standardization.

    Arg:
        atoms   `ase.Atoms` object of the bulk you want to standardize
    Returns:
        standardized_struct     `pymatgen.Structure` of the standardized bulk
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    standardized_struct = sga.get_conventional_standard_structure()
    return standardized_struct


def is_structure_invertible(structure):
    '''
    This function figures out whether or not an `pymatgen.Structure` object has
    symmetricity. In this function, the affine matrix is a rotation matrix that
    is multiplied with the XYZ positions of the crystal. If the z,z component
    of that is negative, it means symmetry operation exist, it could be a
    mirror operation, or one that involves multiple rotations/etc. Regardless,
    it means that the top becomes the bottom and vice-versa, and the structure
    is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Arg:
        structure   A `pymatgen.Structure` object.
    Returns
        A boolean indicating whether or not your `ase.Atoms` object is
        symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    '''
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def flip_struct(struct):
    '''
    Flips an atoms object upside down. Normally used to flip surfaces.

    Arg:
        atoms   `pymatgen.Structure` object
    Returns:
        flipped_struct  The same `ase.Atoms` object that was fed as an
                        argument, but flipped upside down.
    '''
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, 'x', rotate_cell=True, center='COM')
    if atoms.cell[2][2] < 0.:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.wrap()

    flipped_struct = AseAtomsAdaptor.get_structure(atoms)
    return flipped_struct


def tile_atoms(atoms):
    '''
    This function will repeat an atoms structure in the x and y direction until
    the x and y dimensions are at least as wide as the MIN_XY constant.

    Args:
        atoms   `ase.Atoms` object of the structure that you want to tile
    Returns:
        atoms_tiled     An `ase.Atoms` object that's just a tiled version of
                        the `atoms` argument.
    '''
    x_length = np.linalg.norm(atoms.cell[0])
    y_length = np.linalg.norm(atoms.cell[1])
    nx = int(math.ceil(MIN_XY/x_length))
    ny = int(math.ceil(MIN_XY/y_length))
    n_xyz = (nx, ny, 1)
    atoms_tiled = atoms.repeat(n_xyz)
    return atoms_tiled


def tag_surface_atoms(bulk_atoms, surface_atoms):
    '''
    Referencing codes from pymatgen to get a list of surface atoms indices of a
    surface's top surface.  Taken from pymatgen.core.surface Class Slab,
    `get_surface_sites`.  https://pymatgen.org/pymatgen.core.surface.html

    Arg:
        bulk_atoms      `ase.Atoms` format of the respective bulk structure
        surface_atoms   The surface where you are trying to find surface sites in
                        `ase.Atoms` format
    Output:
        indices_list    A list that contains the indices of
                        the surface atoms
    '''
    # Get a dictionary of unique coordination numbers
    # for atoms in a bulk structure.
    # for example, Pt would have cn=3 and cn=12 in Pt bulk
    bulk_struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(bulk_struct)
    sym_struct = sga.get_symmetrized_structure()
    unique_indices = [equ[0] for equ in sym_struct.equivalent_indices]
    voronoi_nn = VoronoiNN()
    bulk_cn_dict = {}
    for idx in unique_indices:
        elem = sym_struct[idx].species_string
        if elem not in bulk_cn_dict.keys():
            bulk_cn_dict[elem] = []
        cn = voronoi_nn.get_cn(sym_struct, idx, use_weights=True)
        cn = float('%.5f' % (round(cn, 5)))
        if cn not in bulk_cn_dict[elem]:
            bulk_cn_dict[elem].append(cn)

    # Determine the surface atoms indices from here
    surface_struct = AseAtomsAdaptor.get_structure(surface_atoms)
    voronoi_nn = VoronoiNN()
    weights = [site.species.weight for site in surface_struct]
    center_of_mass = np.average(surface_struct.frac_coords,
                                weights=weights, axis=0)

    surface_indices = set()
    for idx, site in enumerate(surface_struct):
        if site.frac_coords[2] > center_of_mass[2]:
            try:
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = float('%.5f' % (round(cn, 5)))
                # surface atoms are under-coordinated
                if cn < min(bulk_cn_dict[site.species_string]):
                    surface_indices.append(idx)
            except RuntimeError:
                # or if pathological error is returned,
                # indicating a surface site
                surface_indices.add(idx)

    tags = [1 if i in surface_indices else 0
            for i, _ in enumerate(surface_atoms)]
    surface_atoms.set_tags(tags)


def add_adsorbate_onto_surface(surface_atoms, surface_sites, adsorbate):
    '''
    There are a lot of small details that need to be considered when adding an
    adsorbate onto a surface. This function will take care of those details for
    you.

    Args:
        surface         An `ase.Atoms` object of the surface
        surface_sites   A list of surface atoms' indices of the surface
        adsorbate       An `ase.Atoms` object of the adsorbate

    Returns:
        ads_surface     An `ase graphic Atoms` object containing the adsorbate and
                        surface. The bulk atoms will be tagged with `0`; the
                        surface atoms will be tagged with `1`, and the the
                        adsorbate atoms will be tagged with `2` or above.
    '''
    # convert surface atoms into graphic atoms object
    surface_gratoms = catkit.Gratoms(surface_atoms)
    surface_atom_indices = [i for i, atom in enumerate(surface_atoms)
                            if atom.tag == 1]
    surface_gratoms.set_surface_atoms(surface_atom_indices)
    surface_gratoms.pbc = np.array([True, True, False])

    # set up the adsorbate into graphic atoms object
    # with its connectivity matrix
    adsorbate_gratoms = convert_adsorbate_atoms_to_gratoms(adsorbate['atoms_obj'])

    # generate all possible adsorption configurations on that surface.
    # The "bonds" argument automatically take care of mono vs.
    # bidentate adsorption configuration.
    builder = catkit.gen.adsorption.Builder(surface_gratoms)
    adsorbed_surfaces = builder.add_adsorbate(adsorbate_gratoms,
                                              bonds=adsorbate['bond_idx'],
                                              index=-1)

    # Filter out unreasonable structures.
    # Then pick one from the reasonable configurations list as an output.
    reasonable_adsorbed_surfaces = [surface for surface in adsorbed_surfaces
                                    if is_config_reasonable(surface) is True]
    adsorbed_surface = random.choice(reasonable_adsorbed_surfaces)
    return adsorbed_surface


def get_connectivity(adsorbate):
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


def convert_adsorbate_atoms_to_gratoms(adsorbate):
    """
    Convert adsorbate atoms object into graphic atoms object,
    so the adsorbate can be placed onto the surface with optimal
    configuration. Set tags for adsorbate atoms to 2, to distinguish
    them from surface atoms.

    Args:
        adsorbate           An `ase.Atoms` object of the adsorbate

    Returns:
        adsorbate_gratoms   An graphic atoms object of the adsorbate.
    """
    connectivity = get_connectivity(adsorbate)
    adsorbate_gratoms = catkit.Gratoms(adsorbate, edges=connectivity)
    adsorbate_gratoms.set_tags([2]*len(adsorbate_gratoms))
    return adsorbate_gratoms


def is_config_reasonable(adslab):
    """
    Function that check weather the adsorbate placement
    is reasonable. For any atom in the adsorbate, if the distance
    between the atom and slab atoms are closer than 80% of
    their expected covalent bond, we reject that placement.

    Args:
        adslab          An `ase.Atoms` object of the adsorbate+slab complex.

    Returns:
        A boolean indicating whether or not the adsorbate placement is
        reasonable.
    """
    vnn = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)
    adsorbate_indices = [atom.index for atom in adslab if atom.tag == 2]
    structure = AseAtomsAdaptor.get_structure(adslab)

    # check the covalent radius between each adsorbate atoms
    # and its nearest neighbors that are slab atoms
    for idx in adsorbate_indices:
        nearneighbors = vnn.get_nn_info(structure, n=idx)
        slab_nn = [nn for nn in nearneighbors if nn['site_index'] not in adsorbate_indices]
        for nn in slab_nn:
            ads_elem = structure[idx].species_string
            nn_elem = structure[nn['site_index']].species_string
            cov_bond_thres = 0.8 * (covalent_radius[ads_elem] + covalent_radius[nn_elem])/100
            actual_dist = adslab.get_distance(idx, nn['site_index'], mic=True)
            if actual_dist < cov_bond_thres:
                return False
    return True


def constrain_surface(atoms):
    '''
    This function fixes sub-surface atoms of a surface. Also works on systems
    that have surface + adsorbate(s), as long as the bulk atoms are tagged with
    `0`, surface atoms are tagged with `1`, and the adsorbate atoms are tagged
    with `2` or above

    Inputs:
        atoms           `ase.Atoms` class of the surface system. The tags of
                        these atoms must be set such that any bulk/surface
                        atoms are tagged with `0` or `1`, resectively, and any
                        adsorbate atom is tagged with a 2 or above.
    Returns:
        atoms           A deep copy of the `atoms` argument, but where the appropriate
                        atoms are constrained.
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = [True if atom.tag == 0 else False for atom in atoms]
    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms
