from shutil import which

from ase.calculators.orca import ORCA, OrcaProfile
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
    "DEFGRID3",
    "ALLPOP",
    "NBO",
]
ORCA_BLOCKS = [
    "%scf Convergence Tight maxiter 300 end",
    "%elprop Dipole true Quadrupole true end",
    '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end',
    '%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 end',
]
ORCA_ASE_SIMPLE_INPUT = " ".join([ORCA_FUNCTIONAL] + [ORCA_BASIS] + ORCA_SIMPLE_INPUT)
OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
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

    MyOrcaProfile = OrcaProfile([which("orca")])
    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=output_directory,
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])
