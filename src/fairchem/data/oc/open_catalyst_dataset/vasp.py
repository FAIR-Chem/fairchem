'''
This submodule contains the scripts that the we used to run VASP.

Note that some of these scripts were taken and modified from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of authors.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import os
import numpy as np
import ase.io
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.vasp import Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC


VASP_FLAGS = {'ibrion': 2,
              'nsw': 200,
              'isif': 0,
              'isym': 0,
              'lreal': 'Auto',
              'ediffg': -0.03,
              'symprec': 1e-10,
              'encut': 350.,
              'pp_version': '5.4',
              'gga': 'RP',
              'pp': 'PBE',
              'xc': 'PBE'}

# TODO:  Sid/Caleb to put the location of the directory of pseudopotentials
VASP_PP_PATH = ''


def run_vasp(atoms, vasp_flags=None):
    '''
    Will relax the input atoms given the VASP flag inputs.

    Args:
        atoms       `ase.Atoms` object that we want to relax.
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator. Defaults to a standerd set of values if `None`
    Returns:
    '''
    if vasp_flags is None:  # Immutable default
        vasp_flags = VASP_FLAGS.copy()

    atoms, vasp_flags = _clean_up_inputs(atoms, vasp_flags)
    vasp_flags = _set_vasp_command(vasp_flags)
    trajectory = relax_atoms(atoms, vasp_flags)
    return trajectory


def _clean_up_inputs(atoms, vasp_flags):
    '''
    Parses the inputs and makes sure some things are straightened out.

    Arg:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        atoms       `ase.Atoms` object of the structure we want to relax, but
                    with the unit vectors fixed (if needed)
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    # Check that the unit vectors obey the right-hand rule, (X x Y points in
    # Z). If not, then flip the order of X and Y to enforce this so that VASP
    # is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    # Push the pseudopotentials into the OS environment for VASP to pull from
    os.environ['VASP_PP_PATH'] = VASP_PP_PATH

    # Calculate and set the k points
    k_pts = calculate_surface_k_points(atoms)
    vasp_flags['kpts'] = k_pts

    return atoms, vasp_flags


def calculate_surface_k_points(atoms):
    '''
    For surface calculations, it's a good practice to calculate the k-point
    mesh given the unit cell size. We do that on-the-spot here.

    Arg:
        atoms   `ase.Atoms` object of the structure we want to relax
    Returns:
        k_pts   A 3-tuple of integers indicating the k-point mesh to use
    '''
    cell = atoms.get_cell()
    order = np.inf
    a0 = np.linalg.norm(cell[0], ord=order)
    b0 = np.linalg.norm(cell[1], ord=order)
    multiplier = 40
    k_pts = (max(1, int(round(multiplier/a0))),
             max(1, int(round(multiplier/b0))),
             1)
    return k_pts


def _set_vasp_command(n_processors=16, vasp_executable='vasp_std'):
    '''
    This function assigns the appropriate call to VASP to the `$VASP_COMMAND`
    variable.
    '''
    # TODO:  Sid and/or Caleb to figure out what exactly to put here to make
    # things work. Here are some examples:
    # https://github.com/ulissigroup/GASpy/blob/master/gaspy/vasp_functions.py#L167
    # https://github.com/ulissigroup/GASpy/blob/master/gaspy/vasp_functions.py#L200
    command = 'srun -n %d %s' % (n_processors, vasp_executable)
    os.environ['VASP_COMMAND'] = command
    raise NotImplementedError


def relax_atoms(atoms, vasp_flags):
    '''
    Perform a DFT relaxation with VASP and then write the trajectory to the
    'relaxation.traj' file.

    Args:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        images      A list of `ase.Atoms` that comprise the relaxation
                    trajectory
    '''
    # Run the calculation
    calc = Vasp2(**vasp_flags)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    # Read the trajectory from the output file
    images = []
    for atoms in ase.io.read('vasprun.xml', ':'):
        image = atoms.copy()
        image = image[calc.resort]
        image.set_calculator(SPC(image,
                                 energy=atoms.get_potential_energy(),
                                 forces=atoms.get_forces()[calc.resort]))
        images += [image]

    # Write the trajectory
    with TrajectoryWriter('relaxation.traj', 'a') as tj:
        for atoms in images:
            tj.write(atoms)
    return images
