import os
import pickle

from quacc.recipes.orca.core import ase_relax_job, static_job
from quacc.wflow_tools.customizers import strip_decorator

from omdata.orca.calc import (
    OPT_PARAMETERS,
    ORCA_BASIS,
    ORCA_BLOCKS,
    ORCA_FUNCTIONAL,
    ORCA_SIMPLE_INPUT,
    ORCA_SIMPLE_INPUT_QUACC_IGNORE,
)


def single_point_calculation(
    atoms,
    charge,
    spin_multiplicity,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput=ORCA_SIMPLE_INPUT,
    orcablocks=ORCA_BLOCKS,
    nprocs=12,
    outputdir=os.getcwd(),
):
    """
    Wrapper around QUACC's static job to standardize single-point calculations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    outputdir: str
        Directory to move results to upon completion
    """
    from quacc import SETTINGS

    SETTINGS.RESULTS_DIR = outputdir

    o = strip_decorator(static_job)(
        atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput + ORCA_SIMPLE_INPUT_QUACC_IGNORE,
        orcablocks=orcablocks,
        nprocs=nprocs,
    )
    # TODO: how we want to handle what results to save out and where to store
    # them.
    with open(os.path.join(outputdir, "quacc_results.pkl"), "wb") as f:
        pickle.dump(o, f)


def ase_relaxation(
    atoms,
    charge,
    spin_multiplicity,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput=ORCA_SIMPLE_INPUT,
    orcablocks=ORCA_BLOCKS,
    nprocs=12,
    opt_params=OPT_PARAMETERS,
    outputdir=os.getcwd(),
):
    """
    Wrapper around QUACC's ase_relax_job to standardize geometry optimizations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    opt_params: dict
        Dictionary of optimizer parameters
    outputdir: str
        Directory to move results to upon completion
    """
    from quacc import SETTINGS

    SETTINGS.RESULTS_DIR = outputdir

    o = strip_decorator(ase_relax_job)(
        atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput + ORCA_SIMPLE_INPUT_QUACC_IGNORE,
        orcablocks=orcablocks,
        opt_params=opt_params,
        nprocs=nprocs,
    )
    # TODO: how we want to handle what results to save out and where to store them.
    with open(os.path.join(outputdir, "quacc_results.pkl"), "wb") as f:
        pickle.dump(o, f)
