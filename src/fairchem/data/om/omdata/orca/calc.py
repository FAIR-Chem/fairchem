from pathlib import Path
from shutil import which

from ase.calculators.orca import ORCA, OrcaProfile
from pydantic import Field
from sella import Sella

ORCA_FUNCTIONAL = "wB97M-V"
ORCA_BASIS = "def2-TZVPD"
ORCA_SIMPLE_INPUT = [
    "EnGrad",
    "RIJCOSX",
    "def2/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID2",
    "ALLPOP"
]
ORCA_SIMPLE_INPUT_QUACC_IGNORE = []
ORCA_BLOCKS = ["%scf Convergence Tight maxiter 500 end","%elprop Dipole true Quadrupole true end"]
ORCA_ASE_SIMPLE_INPUT = " ".join([ORCA_FUNCTIONAL] + [ORCA_BASIS] + ORCA_SIMPLE_INPUT)
OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 1000,
}


def write_orca_inputs(
    atoms,
    output_directory,
    charge=0,
    mult=1,
    orcasimpleinput=ORCA_ASE_SIMPLE_INPUT,
    orcablocks=" ".join(ORCA_BLOCKS),
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """

    MyOrcaProfile = OrcaProfile([str(Field(Path(which("orca") or "orca")).default)])
    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=output_directory,
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])
