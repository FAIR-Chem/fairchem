from __future__ import annotations

import os
import subprocess
import tempfile
from shutil import which
from typing import TYPE_CHECKING

import ase.io
import numpy as np
from fairchem.data.oc.core.multi_adsorbate_slab_config import (
    MultipleAdsorbateSlabConfig,
)

if TYPE_CHECKING:
    from fairchem.data.oc.core.adsorbate import Adsorbate
    from fairchem.data.oc.core.ion import Ion
    from fairchem.data.oc.core.slab import Slab
    from fairchem.data.oc.core.solvent import Solvent

from fairchem.data.oc.utils.geometry import (
    BoxGeometry,
    Geometry,
    PlaneBoundTriclinicGeometry,
)

# Code adapted from https://github.com/henriasv/molecular-builder/tree/master


class InterfaceConfig(MultipleAdsorbateSlabConfig):
    """
    Class to represent a solvent, adsorbate, slab, ion config. This class only
    returns a fixed combination of adsorbates placed on the surface. Solvent
    placement is performed by packmol
    (https://m3g.github.io/packmol/userguide.shtml), with the number of solvent
    molecules controlled by its corresponding density. Ion placement is random
    within the desired volume.

    Arguments
    ---------
    slab: Slab
        Slab object.
    adsorbates: List[Adsorbate]
        List of adsorbate objects to place on the slab.
    solvent: Solvent
        Solvent object
    ions: List[Ion] = []
        List of ion objects to place
    num_sites: int
        Number of sites to sample.
    num_configurations: int
        Number of configurations to generate per slab+adsorbate(s) combination.
        This corresponds to selecting different site combinations to place
        the adsorbates on.
    interstitial_gap: float
        Minimum distance, in Angstroms, between adsorbate and slab atoms as
        well as the inter-adsorbate distance.
    vacuum_size: int
        Size of vacuum layer to add to both ends of the resulting atoms object.
    solvent_interstitial_gap: float
        Minimum distance, in Angstroms, between the solvent environment and the
        adsorbate-slab environment.
    solvent_depth: float
        Volume depth to be used to pack solvents inside.
    pbc_shift: float
        Cushion to add to the packmol volume to avoid overlapping atoms over pbc.
    packmol_tolerance: float
        Packmol minimum distance to impose between molecules.
    mode: str
        "random", "heuristic", or "random_site_heuristic_placement".
        This affects surface site sampling and adsorbate placement on each site.

        In "random", we do a Delaunay triangulation of the surface atoms, then
        sample sites uniformly at random within each triangle. When placing the
        adsorbate, we randomly rotate it along xyz, and place it such that the
        center of mass is at the site.

        In "heuristic", we use Pymatgen's AdsorbateSiteFinder to find the most
        energetically favorable sites, i.e., ontop, bridge, or hollow sites.
        When placing the adsorbate, we randomly rotate it along z with only
        slight rotation along x and y, and place it such that the binding atom
        is at the site.

        In "random_site_heuristic_placement", we do a Delaunay triangulation of
        the surface atoms, then sample sites uniformly at random within each
        triangle. When placing the adsorbate, we randomly rotate it along z with
        only slight rotation along x and y, and place it such that the binding
        atom is at the site.

        In all cases, the adsorbate is placed at the closest position of no
        overlap with the slab plus `interstitial_gap` along the surface normal.
    """

    def __init__(
        self,
        slab: Slab,
        adsorbates: list[Adsorbate],
        solvent: Solvent,
        ions: list[Ion] | None = None,
        num_sites: int = 100,
        num_configurations: int = 1,
        interstitial_gap: float = 0.1,
        vacuum_size: int = 15,
        solvent_interstitial_gap: float = 2,
        solvent_depth: float = 8,
        pbc_shift: float = 0.0,
        packmol_tolerance: float = 2,
        mode: str = "random_site_heuristic_placement",
    ):
        super().__init__(
            slab=slab,
            adsorbates=adsorbates,
            num_sites=num_sites,
            num_configurations=num_configurations,
            interstitial_gap=interstitial_gap,
            mode=mode,
        )

        self.solvent = solvent
        self.ions = ions
        self.vacuum_size = vacuum_size
        self.solvent_depth = solvent_depth
        self.solvent_interstitial_gap = solvent_interstitial_gap
        self.pbc_shift = pbc_shift
        self.packmol_tolerance = packmol_tolerance

        self.n_mol_per_volume = solvent.molecules_per_volume

        self.atoms_list, self.metadata_list = self.create_interface_on_sites(
            self.atoms_list, self.metadata_list
        )

    def create_interface_on_sites(
        self, atoms_list: list[ase.Atoms], metadata_list: list[dict]
    ):
        """
        Given adsorbate+slab configurations generated from
        (Multi)AdsorbateSlabConfig and its corresponding metadata, create the
        solvent/ion interface on top of the provided atoms objects.
        """
        atoms_interface_list = []
        metadata_interface_list = []

        for atoms, adsorbate_metadata in zip(atoms_list, metadata_list):
            cell = atoms.cell.copy()
            unit_normal = cell[2] / np.linalg.norm(cell[2])
            cell[2] = self.solvent_depth * unit_normal
            volume = cell.volume
            n_solvent_mols = int(self.n_mol_per_volume * volume)

            if cell.orthorhombic:
                box_length = cell.lengths()
                geometry = BoxGeometry(
                    center=box_length / 2, length=box_length - self.pbc_shift
                )
            else:
                geometry = PlaneBoundTriclinicGeometry(cell, pbc=self.pbc_shift)

            solvent_ions_atoms = self.create_packmol_atoms(geometry, n_solvent_mols)
            solvent_ions_atoms.set_cell(cell)

            max_z = atoms.positions[:, 2].max() + self.solvent_interstitial_gap
            translation_vec = cell[2]
            translation_vec[2] = max_z
            solvent_ions_atoms.translate(translation_vec)

            interface_atoms = atoms + solvent_ions_atoms
            interface_atoms.center(vacuum=self.vacuum_size, axis=2)
            interface_atoms.wrap()

            atoms_interface_list.append(interface_atoms)

            metadata = {
                "adsorbates": adsorbate_metadata,
                "solvent": self.solvent.name,
                "ions": [x.name for x in self.ions],
                "ion_concentrations": [
                    x.get_ion_concentration(volume) for x in self.ions
                ],
                "solvent_depth": self.solvent_depth,
                "solvent_volume": volume,
                "slab_millers": self.slab.millers,
                "slab_shift": self.slab.shift,
                "slab_top": self.slab.top,
                "bulk_idx": self.slab.bulk.bulk_id_from_db,
                "ase_tags": interface_atoms.get_tags(),
            }
            metadata_interface_list.append(metadata)

        return atoms_interface_list, metadata_interface_list

    def create_packmol_atoms(self, geometry: Geometry, n_solvent_mols: int):
        """
        Pack solvent molecules in a provided unit cell volume. Packmol is used
        to randomly pack solvent molecules in the desired volume.

        Arguments:
            geometry (Geometry): Geometry object corresponding to the desired cell.
            n_solvent_mols (int): Number of solvent molecules to pack in the volume.
        """
        cell = geometry.cell
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "out.pdb")
            self.solvent.atoms.write(f"{tmp_dir}/solvent.pdb", format="proteindatabank")

            # When placing a single ion, packmol strangely always places it at
            # the boundary of the cell. This hacky fix manually places
            # the ion in a random location in the cell. Packmol then will fix
            # these atoms and not optimize them during its optimization, only
            # optimizing solvent molecules arround the ion.
            for i, ion in enumerate(self.ions):
                ion_atoms = ion.atoms.copy()
                ion_atoms.set_cell(cell)
                self.randomize_coords(ion_atoms)
                ion_atoms.write(f"{tmp_dir}/ion_{i}.pdb", format="proteindatabank")

            # write packmol input
            packmol_input = os.path.join(tmp_dir, "input.inp")
            with open(packmol_input, "w") as f:
                f.write(f"tolerance {self.packmol_tolerance}\n")
                f.write("filetype pdb\n")
                f.write(f"output {output_path}\n")

                # write solvent
                f.write(
                    geometry.packmol_structure(
                        f"{tmp_dir}/solvent.pdb", n_solvent_mols, "inside"
                    )
                )

                for i in range(len(self.ions)):
                    f.write(f"structure {tmp_dir}/ion_{i}.pdb\n")
                    f.write("  number 1\n")
                    f.write("  fixed 0 0 0 0 0 0\n")
                    f.write("end structure\n\n")

            self.run_packmol(packmol_input)

            solvent_ions_atoms = ase.io.read(output_path, format="proteindatabank")
            solvent_ions_atoms.set_pbc(True)
            solvent_ions_atoms.set_tags([3] * len(solvent_ions_atoms))

        return solvent_ions_atoms

    def run_packmol(self, packmol_input: str):
        """
        Run packmol.
        """
        packmol_cmd = which("packmol")
        if not packmol_cmd:
            raise OSError("packmol not found.")

        ps = subprocess.Popen(
            f"{packmol_cmd} < {packmol_input}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = ps.communicate()
        if err:
            raise OSError(err.decode("utf-8"))

    def randomize_coords(self, atoms: ase.Atoms):
        """
        Randomly place the atoms in its unit cell.
        """
        cell_weights = np.random.rand(3)
        cell_weights /= np.sum(cell_weights)
        xyz = np.dot(cell_weights, atoms.cell)
        atoms.set_center_of_mass(xyz)
