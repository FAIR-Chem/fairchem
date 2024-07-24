from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from fairchem.core.scripts import download_large_files


def is_successful(best_pred_energy, best_dft_energy, SUCCESS_THRESHOLD=0.1):
    """
    Computes the success rate given the best predicted energy
    and the best ground truth DFT energy.

    success_parity: The standard definition for success, where ML needs to be
    within the SUCCESS_THRESHOLD, or lower, of the DFT energy.

    Returns: Bool
    """
    # Given best ML and DFT energy, compute various success metrics:
    # success_parity: base success metric (ML - DFT <= SUCCESS_THRESHOLD)
    diff = best_pred_energy - best_dft_energy
    return diff <= SUCCESS_THRESHOLD


def compute_valid_ml_success(ml_data, dft_data):
    """
    Computes validated ML success rates.
    Here, results are generated only from ML. DFT single-points are used to
    validate whether the ML energy is within 0.1eV of the DFT energy of the
    predicted structure. If valid, the ML energy is compared to the ground
    truth DFT energy, otherwise it is discarded.

    Return validated ML success rates.
    """

    success_rate = 0.0

    for system in dft_data:
        # For `system`, collect all ML adslabs and their corresponding energies
        ml_adslabs, ml_energies = [], []
        for config in ml_data[system]:
            ml_adslabs.append(config)
            ml_energies.append(ml_data[system][config]["ml_energy"])

        min_ml_idx = np.argmin(ml_energies)
        min_adslab = ml_adslabs[min_ml_idx]
        best_ml_energy = ml_energies[min_ml_idx]
        # If the best ML energy is not within 0.1eV
        # of its DFT energy evaluation, discard.
        ml_dft_energy = ml_data[system][min_adslab]["ml+dft_energy"]
        diff = abs(ml_dft_energy - best_ml_energy)
        if diff > 0.1:
            continue

        best_dft_energy = min(list(dft_data[system].values()))

        success = is_successful(best_ml_energy, best_dft_energy)
        success_rate += success

    success_rate /= len(dft_data)

    print("=" * 50)
    print(f"Success Rate (%): {100*success_rate}")


def get_dft_data(targets):
    """
    Organizes the released target mapping for evaluation lookup.

    Returns: Dict:
        {
           'system_id 1': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
           'system_id 2': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
           ...
        }
    """
    dft_data = defaultdict(dict)
    for system in targets:
        for adslab in targets[system]:
            dft_data[system][adslab[0]] = adslab[1]

    return dft_data


def process_ml_data(results_file, model, metadata, ml_dft_targets, dft_data):
    """
    For ML systems in which no configurations made it through the physical
    constraint checks, set energies to an arbitrarily high value to ensure
    a failure case in evaluation.

    Returns: Dict:
        {
           'system_id 1': {'config_id 1': {'ml_energy': predicted energy, 'ml+dft_energy': dft energy of ML structure} ...},
           'system_id 2': {'config_id 1': {'ml_energy': predicted energy, 'ml+dft_energy': dft energy of ML structure} ...},
           ...
        }
    """
    preds = np.load(results_file)
    ml_data = defaultdict(dict)

    for _id, energy in zip(preds["ids"], preds["energy"]):
        sid, _ = _id.split("_")

        info = metadata[int(sid)]
        sysid = info["system_id"]
        config = info["config_id"]

        ml_dft_energy = ml_dft_targets[model][sysid][config]
        ml_data[sysid][config] = {"ml_energy": energy, "ml+dft_energy": ml_dft_energy}

    # set missing systems to high energy
    # set missing systems to 0 DFT compute
    for system in dft_data:
        if system not in ml_data:
            ml_data[system] = defaultdict(dict)
        for config in dft_data[system]:
            if config not in ml_data[system]:
                _dict = {
                    "ml_energy": 1e10,
                    "ml+dft_energy": 1e10,
                }
                ml_data[system][config] = _dict

    # for ML systems with no available ml+dft datapoints, set to an arbitrarily
    # high energy value
    for system in ml_data:
        for config in ml_data[system]:
            if not ml_data[system][config]["ml+dft_energy"]:
                ml_data[system][config]["ml+dft_energy"] = 1e10

    return ml_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemnet-oc-2M", "escn-2M", "scn-2M"],
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to predictions to evaluate. NPZ format.",
    )

    return parser.parse_args()


def main():
    """
    This script takes in your prediction file (npz format)
    and the ML model name used for ML relaxations.
    Then using a mapping file, dft ground truth energy,
    and ML relaxed dft energy returns the success rate of your predictions.
    """

    args = parse_args()

    # targets and metadata are expected to be in
    # the same directory as this script
    if (
        not Path(__file__).with_name("oc20dense_val_targets.pkl").exists()
        or not Path(__file__).with_name("ml_relaxed_dft_targets.pkl").exists()
    ):
        download_large_files.download_file_group("adsorbml")
    targets = pickle.load(
        open(Path(__file__).with_name("oc20dense_val_targets.pkl"), "rb")
    )
    ml_dft_targets = pickle.load(
        open(Path(__file__).with_name("ml_relaxed_dft_targets.pkl"), "rb")
    )
    metadata = pickle.load(
        open(Path(__file__).with_name("oc20dense_mapping.pkl"), "rb")
    )

    ###### Process DFT Data ######
    dft_data = get_dft_data(targets)

    ###### Process ML Data ######
    ml_data = process_ml_data(
        args.results_file, args.model, metadata, ml_dft_targets, dft_data
    )

    ###### Compute Metrics ######
    print(f"Prediction file: {args.results_file}")
    compute_valid_ml_success(ml_data, dft_data)


if __name__ == "__main__":
    main()
