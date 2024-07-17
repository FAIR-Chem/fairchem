from __future__ import annotations

from pathlib import Path

import ase
from ase.calculators.vasp import Vasp


def setup_vasp_calc_mof(atoms: ase.Atoms, path: Path):
    """
    Create a VASP calculator for MOF relaxation and write VASP input files to path.
    """
    # Setting number of k-points to 1x1x1. Increase number of k-points
    # if unit cell size is too small.
    kpoints = ((1, 1, 1),)
    calc = Vasp(
        nwrite=2,
        istart=0,
        gga="PE",
        ivdw=12,
        encut=600.0,
        lcharg=False,
        lwave=False,
        ismear=0,
        sigma=0.2,
        ispin=2,
        prec="Accurate",
        nelm=60,
        nelmin=2,
        ediff=1e-5,
        ediffg=-0.05,
        maxmix=40,
        nsw=2000,
        ibrion=2,
        isif=3,  # Relax atom positions & unit cell parameters
        potim=0.01,
        algo="NORMAL",
        ldiag=True,
        lreal="Auto",
        lplane=True,
        ncore=4,
        kpts=kpoints,
        gamma=True,
        isym=0,
        directory=path,
    )
    atoms.set_calculator(calc)
    calc.write_input(
        atoms,
        properties=("energy",),
        system_changes=tuple(ase.calculators.calculator.all_changes),
    )


def setup_vasp_mof_and_ads(atoms: ase.Atoms, path: Path):
    """
    Create a VASP calculator for MOF + Adsorbate(s) relaxation and write VASP input files to path.
    For these relaxations, the MOF has already been pre-relaxed.
    """
    # Setting number of k-points to 1x1x1. Increase number of k-points
    # if unit cell size is too small.
    kpoints = ((1, 1, 1),)
    calc = Vasp(
        nwrite=2,
        istart=0,
        gga="PE",
        ivdw=12,
        encut=600.0,
        lcharg=False,
        lwave=False,
        ismear=0,
        sigma=0.2,
        ispin=2,
        prec="Accurate",
        nelm=60,
        nelmin=2,
        ediff=1e-5,
        ediffg=-0.05,
        maxmix=40,
        nsw=2000,
        ibrion=2,
        isif=2,  # Relax atom positions only
        potim=0.01,
        algo="NORMAL",
        ldiag=True,
        lreal="Auto",
        lplane=True,
        ncore=4,
        kpts=kpoints,
        gamma=True,
        isym=0,
        directory=path,
    )
    atoms.set_calculator(calc)
    calc.write_input(
        atoms,
        properties=("energy",),
        system_changes=tuple(ase.calculators.calculator.all_changes),
    )


if __name__ == "__main__":
    import os

    import ase.io

    os.environ["VASP_PP_PATH"] = "vasp_pp"  # Path to Vasp Pseudo Potentials

    # MOF-only relaxation
    mof = ase.io.read("ADOCEC.cif")
    setup_vasp_calc_mof(mof, Path("vasp_mof"))

    # MOF + Ads relaxation
    mof_ads = ase.io.read("ADOCEC_ads.cif")
    setup_vasp_mof_and_ads(mof_ads, Path("vasp_mof_ads"))
