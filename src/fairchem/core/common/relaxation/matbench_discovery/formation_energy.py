"""
Stability metrics for predicted energies of WBM structures as implemented in matbench-discovery.

This is an implementation of
https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/energy.py and
using our lmdbs to evaluate, see wbm2aselmdb.py for info on WBM lmdbs
"""
from pathlib import Path
import logging
from collections import Counter
from tqdm import trange

import torch
from monty.serialization import loadfn
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDEntry


# threshold on hull distance for a material to be considered stable
# set to same as https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/__init__.py
STABILITY_THRESHOLD = 0
MDB_ROOT = Path("/data/shared/matbench-discovery/")
try:
    # try to load them
    mp_elemental_ref_energies = loadfn(MDB_ROOT / "mp_elem_ref_entries.json")
except FileNotFoundError:
    logging.warn(
        f"Unable to load mp elemental reference energies from {MDB_ROOT / 'mp_elem_ref_entries.json'}\n"
        "Will try to download them using matbench-discovery"
    )
    try:
        # try to download from MDB (needs to be installed)
        import pandas as pd
        from pymatgen.entries.computed_entries import ComputedEntry
        from matbench_discovery.data import DataFiles
        mp_elem_ref_entries = (
            pd.read_json(DataFiles.mp_elemental_ref_entries.path, typ="series")
            .map(ComputedEntry.from_dict)
            .to_dict()
        )
    except ImportError as err:
        raise ImportError(
            "Unable to download mp elemental reference energies, matbench-discovery is not installed"
        ) from err


def get_formation_energy_per_atom(
    relaxed_energy: torch.Tensor,
    atomic_numbers: tuple[torch.Tensor],
    elemental_ref_energies: dict[str, float] = mp_elemental_ref_energies,
) -> torch.Tensor:
    """Get formation energy for a phase diagram entry (1st arg, composition + absolute
    energy) and a dict mapping elements to per-atom reference energies (2nd arg).

    Args:
        relaxed_energy: tensor of relaxed energies
        atomic_numbers: tuple of tensors with atomic numbers for each predicted energy
        elemental_ref_energies (dict[str, float], optional): Must be a covering set (for
            entry) of terminal reference energies, i.e. eV/atom of the lowest energy
            elemental phase for each element. Defaults to MP elemental reference
            energies as collected on 2022-09-19 get_elemental_ref_entries(). This was
            tested to give the same formation energies as found in MP.

    Returns:
        tensor: tensor of computed formation energies per atom
    """

    formation_energy = torch.empty_like(relaxed_energy)
    for idx in trange(len(relaxed_energy), desc="Computing formation energy from relaxed energy."):
        comp = Composition(Counter(atomic_numbers[idx]))
        energy = relaxed_energy[idx]
        try:
            e_refs = {str(el): elemental_ref_energies[str(el)] for el in comp}

            for key, ref_entry in e_refs.items():
                if isinstance(ref_entry, dict):
                    e_refs[key] = PDEntry.from_dict(ref_entry)

            e_form = energy - sum(comp[el] * e_refs[str(el)].energy_per_atom for el in comp)
            formation_energy[idx] = e_form / comp.num_atoms
        except (TypeError, Exception):
            formation_energy[idx] = torch.nan
            logging.warning(f"Failed to compute formation energy for {comp=}")

    return formation_energy
