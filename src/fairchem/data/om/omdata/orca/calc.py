from enum import Enum
from shutil import which

from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from sella import Sella

# ECP sizes taken from Table 6.5 in the Orca 5.0.3 manual
ECP_SIZE = {
    **{i: 28 for i in range(37, 55)},
    **{i: 46 for i in range(55, 58)},
    **{i: 28 for i in range(58, 72)},
    **{i: 60 for i in range(72, 87)},
}

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
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 end",
    '%basis GTOName "def2-tzvpd.bas" end',
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


class Vertical(Enum):
    Default = "default"
    MetalOrganics = "metal-organics"


def get_symm_break_block(atoms: Atoms, charge: int) -> str:
    """
    Determine the ORCA Rotate block needed to break symmetry in a singlet

    This is determined by taking the sum of atomic numbers less any charge (because
    electrons are negatively charged) and removing any electrons that are in an ECP
    and dividing by 2. This gives the number of occupied orbitals, but since ORCA is
    0-indexed, it gives the index of the LUMO.

    We use a rotation angle of 20 degrees or about a 12% mixture of LUMO into HOMO.
    This is somewhat arbitrary but similar to the default setting in Q-Chem, and seemed
    to perform well in tests of open-shell singlets.
    """
    n_electrons = sum(atoms.get_atomic_numbers()) - charge
    ecp_electrons = sum(
        ECP_SIZE.get(at_num, 0) for at_num in atoms.get_atomic_numbers()
    )
    n_electrons -= ecp_electrons
    lumo = n_electrons // 2
    return f"%scf rotate {{{lumo-1}, {lumo}, 20, 1, 1}} end end"


def write_orca_inputs(
    atoms: Atoms,
    output_directory,
    charge: int = 0,
    mult: int = 1,
    orcasimpleinput: str = ORCA_ASE_SIMPLE_INPUT,
    orcablocks: str = " ".join(ORCA_BLOCKS),
    vertical=Vertical.Default,
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """

    MyOrcaProfile = OrcaProfile([which("orca")])
    if vertical == Vertical.MetalOrganics and mult == 1:
        orcasimpleinput += " UKS"
        orcablocks += f" {get_symm_break_block(atoms, charge)}"

    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=output_directory,
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])
