import glob
import os

from pymatgen.io.vasp.outputs import *

"""
This script provides utility functions that can be useful for trying to
evaluate AdsorbML, or similar, pipelines.

NOTE - VASP was used for all DFT calculations and the scripts provided expect
VASP output files. While other DFT tools can be used, the following scripts
would need to be modified.
"""


def converged_oszicar(path, nelm=60, ediff=1e-4, idx=0):
    """
    --- FOR VASP USERS ---

    Given a folder containing DFT outputs, ensures the system has converged
    electronically.

    args:
        path: Path to DFT outputs.
        nelm: Maximum number of electronic steps used.
        ediff: Energy difference condition for terminating the electronic loop.
        idx: Frame to check for electronic convergence. 0 for SP, -1 for RX.
    """
    try:
        system = os.path.join(path, "OSZICAR")
        oszicar = Oszicar(system)
        estep = oszicar.electronic_steps[idx]

        # if scf stops before NELM, valid point
        if len(estep) < nelm:
            return True
        # verify the last scf step is not valid
        if abs(estep[-1]["dE"]) < ediff:
            return True

        return False
    except:
        # systems with OSZICARs that cannot be parsed are treated as failures.
        return False


def count_scf(path):
    """
    --- FOR VASP USERS ---

    Given a folder containing DFT outputs, compute total ionic and SCF steps

    args:
        path: Path to DFT outputs.

    return:
        scf_steps (int): Total number of electronic steps performed.
        ionic_steps (int): Total number of ionic steps performed.
    """
    oszicar_files = glob.glob(os.path.join(path, "OSZICAR*"))
    scf_steps = 0
    ionic_steps = 0

    for fname in oszicar_files:
        try:
            oszicar = Oszicar(fname)
            ionic_steps += len(oszicar.electronic_steps)
            for step in oszicar.electronic_steps:
                scf_steps += len(step)
        except:
            continue

    return scf_steps, ionic_steps
