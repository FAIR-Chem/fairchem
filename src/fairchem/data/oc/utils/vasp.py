"""
This submodule contains the scripts that the we used to run VASP.

Note that some of these scripts were taken and modified from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of authors.
"""

from __future__ import annotations

__author__ = "Kevin Tran"
__email__ = "ktran@andrew.cmu.edu"


import numpy as np
from ase.calculators.vasp import Vasp
from fairchem.data.oc.utils.vasp_flags import VASP_FLAGS


def _clean_up_inputs(atoms, vasp_flags):
    """
    Parses the inputs and makes sure some things are straightened out.

    Arg:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp`
                    calculator
    Returns:
        atoms       `ase.Atoms` object of the structure we want to relax, but
                    with the unit vectors fixed (if needed)
        vasp_flags  A modified version of the 'vasp_flags' argument
    """
    # Make a copy of the vasp_flags so we don't modify the original
    vasp_flags = vasp_flags.copy()
    # Check that the unit vectors obey the right-hand rule, (X x Y points in
    # Z). If not, then flip the order of X and Y to enforce this so that VASP
    # is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    # Calculate and set the k points
    if "kpts" not in vasp_flags:
        k_pts = calculate_surface_k_points(atoms)
        vasp_flags["kpts"] = k_pts

    return atoms, vasp_flags


def calculate_surface_k_points(atoms):
    """
    For surface calculations, it's a good practice to calculate the k-point
    mesh given the unit cell size. We do that on-the-spot here.

    Arg:
        atoms   `ase.Atoms` object of the structure we want to relax
    Returns:
        k_pts   A 3-tuple of integers indicating the k-point mesh to use
    """
    cell = atoms.get_cell()
    order = np.inf
    a0 = np.linalg.norm(cell[0], ord=order)
    b0 = np.linalg.norm(cell[1], ord=order)
    multiplier = 40
    return (
        max(1, int(round(multiplier / a0))),
        max(1, int(round(multiplier / b0))),
        1,
    )


def write_vasp_input_files(
    atoms, outdir=".", vasp_flags=None, pp_setups="minimal", pp_env="VASP_PP_PATH"
):
    """
    Effectively goes through the same motions as the `run_vasp` function,
    except it only writes the input files instead of running.

    Args:
        atoms       `ase.Atoms` object that we want to relax.
        outdir      A string indicating where you want to save the input files.
                    Defaults to '.'
        vasp_flags  A dictionary of settings we want to pass to the `Vasp`
                    calculator. Defaults to a standerd set of values if `None`
        pp_setups   Pseudopotential setups to use - https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/setups.py
        pp_env      Environment variable to read for pseudopotentials.
    """
    if vasp_flags is None:  # Immutable default
        vasp_flags = VASP_FLAGS

    atoms, vasp_flags = _clean_up_inputs(atoms, vasp_flags.copy())
    calc = Vasp(directory=outdir, **vasp_flags)
    calc.VASP_PP_PATH = pp_env
    calc.input_params["setups"] = pp_setups
    calc.write_input(atoms)
