import os

import ase.io
from ase.build import add_adsorbate, fcc100, molecule
from ase.constraints import FixAtoms


def main():
    """
    Checks whether ASE's rattle modifies fixed atoms.
    '"""
    # Constructs test system
    slab = fcc100("Cu", size=(3, 3, 3))
    ads = molecule("CO")
    add_adsorbate(slab, ads, 4, offset=(1, 1))
    fix_mask = [atom.index for atom in slab if (atom.tag == 2 or atom.tag == 3)]
    free_mask = [atom.index for atom in slab if (atom.tag != 2 and atom.tag != 3)]
    # Apply constraint to fix the bottom 2 layers of the slab.
    cons = FixAtoms(fix_mask)
    slab.set_constraint(cons)

    original_positions = slab.positions

    ### Rattle system
    rattled_image = slab.copy()
    rattled_image.rattle(stdev=1, seed=23794)

    rattled_positions = rattled_image.positions

    assert (
        original_positions[fix_mask].all() == rattled_positions[fix_mask].all()
    ), "Fixed atoms have been rattled!"

    assert (
        original_positions[free_mask].all() != rattled_positions[free_mask].all()
    ), "Remaining atoms not rattled!"

    print("Test passed! rattle() does not modify fixed atoms")


if __name__ == "__main___":
    main()
