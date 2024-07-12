"""
This script processes ML relaxations and sets it up for the next step.
- Reads final energy and structure for each relaxation
- Filters out anomalies
- Groups together all configurations for one adsorbate-surface system
- Sorts configs by lowest energy first

The following files are saved out:
- cache_sorted_byE.pkl: dict going from the system ID (bulk, surface, adsorbate)
    to a list of configs and their relaxed structures, sorted by lowest energy first.
    This is later used by write_top_k_vasp.py.
- anomalies_by_sid.pkl: dict going from integer sid to boolean representing
    whether it was an anomaly. Anomalies are already excluded from cache_sorted_byE.pkl
    and this file is only used for extra analyses.
- errors_by_sid.pkl: any errors that occurred
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
from collections import defaultdict

import numpy as np
from ase.io import read
from fairchem.data.oc.utils.flag_anomaly import DetectTrajAnomaly
from tqdm import tqdm

SURFACE_CHANGE_CUTOFF_MULTIPLIER = 1.5
DESORPTION_CUTOFF_MULTIPLIER = 1.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process ml relaxations and group them by adsorbate-surface system"
    )
    parser.add_argument(
        "--ml-trajs-path",
        type=str,
        required=True,
        help="ML relaxation trajectories folder path",
    )
    parser.add_argument(
        "--outdir", type=str, default="cache", help="Output directory path"
    )
    parser.add_argument(
        "--workers", type=int, default=80, help="Number of workers for multiprocessing"
    )
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument(
        "--metadata", type=str, help="Path to mapping of sid to metadata"
    )
    parser.add_argument("--surface-dir", type=str, help="Path to surface DFT outputs")

    return parser.parse_args()


def min_diff(atoms_init, atoms_final):
    # used to compare atom positions, taking PBC into account
    positions = atoms_final.positions - atoms_init.positions
    fractional = np.linalg.solve(atoms_init.get_cell(complete=True).T, positions.T).T

    for i, periodic in enumerate(atoms_init.pbc):
        if periodic:
            # Yes, we need to do it twice.
            # See the scaled_positions.py test.
            fractional[:, i] %= 1.0
            fractional[:, i] %= 1.0

    fractional[fractional > 0.5] -= 1
    return np.matmul(fractional, atoms_init.get_cell(complete=True))


def process_mlrs(arg):
    # for each ML trajectory, run anomaly detection and get relaxed energy
    sid, metadata = arg
    system_id = metadata["system_id"]
    adslab_idx = metadata["config_id"]

    try:
        traj = read(f"{args.ml_trajs_path}/{sid}.traj", ":")
        init_atoms, final_atoms = traj[0], traj[-1]
        if fmax:
            for atoms in traj:
                forces = atoms.get_forces()
                tags = atoms.get_tags()
                # only evaluate fmax on free atoms
                free_atoms = [idx for idx, tag in enumerate(tags) if tag != 0]
                _fmax = max(np.sqrt((forces[free_atoms] ** 2).sum(axis=1)))
                if _fmax <= fmax:
                    final_atoms = atoms
                    break

        final_energy = final_atoms.get_potential_energy()
    except:
        error_msg = f"Error parsing traj: {sid}.traj"
        return [sid, system_id, adslab_idx, None, None, True, error_msg]

    surface_id = system_id + "_surface.traj"
    dft_slab_path = os.path.join(SURFACE_DIR, system_id, surface_id)
    if not os.path.isfile(dft_slab_path):
        error_msg = f"Surface {surface_id} unavailable."
        return [sid, system_id, adslab_idx, None, None, True, error_msg]

    slab_traj = read(dft_slab_path, ":")
    tags = init_atoms.get_tags()
    assert sum(tags) > 0  # make sure tag info exists

    # Verify adslab and slab are ordered consistently before anomaly detection
    # This checks that the positions of the initial adslab and clean surface
    # are approximately equivalent.
    diff = abs(min_diff(init_atoms[tags != 2], slab_traj[0])).sum()
    # ML trajectories are saved out after 1 optimization step, so some movement
    # is expected. A cushion of 0.5A is used based off the maximum difference
    # previously measured for sample trajectories.
    assert diff < 0.5

    detector = DetectTrajAnomaly(
        init_atoms,
        final_atoms,
        atoms_tag=tags,
        final_slab_atoms=slab_traj[-1],
        surface_change_cutoff_multiplier=SURFACE_CHANGE_CUTOFF_MULTIPLIER,
        desorption_cutoff_multiplier=DESORPTION_CUTOFF_MULTIPLIER,
    )
    anom = (
        detector.is_adsorbate_dissociated()
        or detector.is_adsorbate_desorbed()
        or detector.has_surface_changed()
    )

    return [sid, system_id, adslab_idx, final_energy, final_atoms, anom, None]


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global fmax, METADATA, SURFACE_DIR
    fmax = args.fmax
    METADATA = args.metadata
    SURFACE_DIR = args.surface_dir

    metadata_by_sid = pickle.load(open(METADATA, "rb"))
    mp_args = list(metadata_by_sid.items())
    pool = mp.Pool(args.workers)
    print("Processing ML trajectories...")
    results = list(tqdm(pool.imap(process_mlrs, mp_args), total=len(mp_args)))

    # process each individual trajectory
    grouped_configs = defaultdict(list)
    anomalies = {}
    errored_sysids = {}
    for result in tqdm(results):
        sid, system, adslab_idx, predE, mlrs, anomaly, error_msg = result
        if predE is None or mlrs is None:
            errored_sysids[sid] = (system, adslab_idx, error_msg)
            continue
        anomalies[sid] = anomaly
        if not anomaly:
            grouped_configs[system].append((adslab_idx, predE, mlrs))

    # group configs by system and sort
    sorted_grouped_configs = {}
    for system, lst in grouped_configs.items():
        sorted_lst = sorted(lst, key=lambda x: x[1])
        sorted_grouped_configs[system] = [(x[0], x[2]) for x in sorted_lst]

    pickle.dump(
        sorted_grouped_configs,
        open(f"{args.outdir}/cache_sorted_byE.pkl", "wb"),
    )
    pickle.dump(anomalies, open(f"{args.outdir}/anomalies_by_sid.pkl", "wb"))
    pickle.dump(errored_sysids, open(f"{args.outdir}/errors_by_sid.pkl", "wb"))
