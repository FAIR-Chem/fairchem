"""
AdsorbML evaluation script. This script expects the results-file to be
organized in a very specific structure in order to evaluate successfully.

Results are to be saved out in a dictionary pickle file, where keys are the
`system_id` and the values are energies and compute information for a
specified `config_id`. For each `config_id` that successfully passes the
physical constraints defined in the manuscript, the following information must
be provided:

    ml_energy: The ML predicted adsorption energy on that particular `config_id`.

    ml+dft_energy: The DFT adsorption energy (SP or RX) as evaluated on
    the predicted ML `config_id` structure. Do note use raw DFT energies,
    ensure these are referenced correctly. None if not available.

    scf_steps: Total number of SCF steps involved in determining the DFT
    adsorption energy on the predicted ML `config_id`. For relaxation
    methods (ML+RX), sum all SCF steps across all frames. 0 if not
    available.

    ionic_steps: Total number of ionic steps in determining the DFT
    adsorption energy on the predicted ML `config_id`. This will be 1 for
    single-point methods (ML+SP). 0 if not available.

NOTE - It is possible that due to the required filtering of physical
constraints, no configurations are valid for a particular `system_id`. In
this case the  system or config id can be excluded entirely from the
results file and will be treated as a failure point at evaluation time.

e.g.
    {
        "6_1134_23":
            {
                "rand11": {
                    "ml_energy": -1.234,
                    "ml+dft_energy": -1.456,
                    "scf_steps": 33,
                    "ionic_steps": 1,
                },
                "rand5": {
                    "ml_energy": -2.489,
                    "ml+dft_energy": -2.109,
                    "scf_steps": 16,
                    "ionic_steps": 1,
                },
                .
                .
                .
            },
        "7_6566_62" :
            {
                "rand79": {
                    "ml_energy": -1.234,
                    "ml+dft_energy": -1.456,
                    "scf_steps": 33,
                    "ionic_steps": 1,
                },
                .
                .
                .

            },
        .
        .
        .
    }
"""

import argparse
import pickle
from collections import defaultdict

import numpy as np

SUCCESS_THRESHOLD = 0.1


def is_successful(best_ml_dft_energy, best_dft_energy):
    """
    Computes the success rate given the best ML+DFT energy and the best ground
    truth DFT energy.


    success_parity: The standard definition for success, where ML needs to be
    within the SUCCESS_THRESHOLD, or lower, of the DFT energy.

    success_much_better: A system in which the ML energy is predicted to be
    much lower (less than the SUCCESS_THRESHOLD) of the DFT energy.
    """
    # Given best ML and DFT energy, compute various success metrics:
    # success_parity: base success metric (ML - DFT <= SUCCESS_THRESHOLD)
    # success_mu: much better than ground truth (ML < DFT - SUCCESS_THRESHOLD)
    diff = best_ml_dft_energy - best_dft_energy
    success_parity = diff <= SUCCESS_THRESHOLD
    success_much_better = diff < -SUCCESS_THRESHOLD

    return success_parity, success_much_better


def compute_hybrid_success(ml_data, dft_data, k):
    """
    Computes AdsorbML success rates at varying top-k values.
    Here, results are generated for the hybrid method, where the top-k ML
    energies are used to to run DFT on the corresponding ML structures. The
    resulting energies are then compared to the ground truth DFT energies.

    Return success rates and DFT compute usage at varying k.
    """
    success_rates = {kk: [0.0, 0.0] for kk in k}
    ml_dft_calls = {kk: {"scf": 0.0, "ionic": 0.0} for kk in k}

    for system in dft_data:
        # For `system`, collect all ML adslabs and their corresponding energies
        ml_adslabs, ml_energies = [], []
        for config in ml_data[system]:
            ml_adslabs.append(config)
            ml_energies.append(ml_data[system][config]["ml_energy"])

        # len(ml_adslabs) and len(dft_data[system]) are not necessarily the same.
        # For a particular placement, ML may have not resulted in an anomaly
        # while DFT may have.
        for kk in success_rates:
            # For each k value, collect the top k systems
            # ML_data is insertion sorted created from a sorted list,
            # gauranteeing the first k are the lowest energies.
            sorted_site_idx = ml_adslabs[:kk]
            # Sanity check to ensure they are sorted
            if len(ml_energies) > kk:
                assert max(ml_energies[:kk]) <= min(ml_energies[kk:])

            best_dft_energy = min(list(dft_data[system].values()))
            # For the top k systems, look up their corresponding ML-DFT
            # energies from `ml_dft_data` to be used for evaluation.
            ml_dft_energies_topk = []
            # Track the number of DFT calls associated with that system.
            for config in sorted_site_idx:
                ml_dft_energies_topk.append(ml_data[system][config]["ml+dft_energy"])
                ml_dft_calls[kk]["scf"] += ml_data[system][config]["scf_steps"]
                ml_dft_calls[kk]["ionic"] += ml_data[system][config]["ionic_steps"]

            best_ml_dft_energy = min(ml_dft_energies_topk)

            success, much_better = is_successful(best_ml_dft_energy, best_dft_energy)
            success_rates[kk][0] += success
            success_rates[kk][1] += much_better

    for kk in success_rates:
        success_rates[kk][0] /= len(dft_data)
        success_rates[kk][1] /= len(dft_data)

    for kk in success_rates:
        print(f"Top-k = {kk}")
        print("=" * 50)
        print(f"Success Rate (%): {100*success_rates[kk][0]}")
        print(
            f"DFT Speedup (SCF): {total_ground_truth_scf_calls/ml_dft_calls[kk]['scf']}"
        )
        print(
            f"DFT Speedup (Ionic): {total_ground_truth_ionic_calls/ml_dft_calls[kk]['ionic']}\n"
        )

    return success_rates, ml_dft_calls


def compute_valid_ml_success(ml_data, dft_data):
    """
    Computes validated ML success rates.
    Here, results are generated only from ML. DFT single-points are used to
    validate whether the ML energy is within 0.1eV of the DFT energy of the
    predicted structure. If valid, the ML energy is compared to the ground
    truth DFT energy, otherwise it is discarded.

    Return validated ML success rates.
    """

    success_rates = [0.0, 0.0]

    for system in dft_data:
        # For `system`, collect all ML adslabs and their corresponding energies
        ml_adslabs, ml_energies = [], []
        for config in ml_data[system]:
            ml_adslabs.append(config)
            ml_energies.append(ml_data[system][config]["ml_energy"])

        min_ml_idx = np.argmin(ml_energies)
        min_adslab = ml_adslabs[min_ml_idx]
        best_ml_energy = ml_energies[min_ml_idx]
        # If the best ML energy is not within 0.1eV of its DFT energy evaluation, discard.
        ml_dft_energy = ml_data[system][min_adslab]["ml+dft_energy"]
        diff = abs(ml_dft_energy - best_ml_energy)
        if diff > 0.1:
            continue

        best_dft_energy = min(list(dft_data[system].values()))

        success, much_better = is_successful(best_ml_energy, best_dft_energy)
        success_rates[0] += success
        success_rates[1] += much_better

    success_rates[0] /= len(dft_data)
    success_rates[1] /= len(dft_data)

    print("=" * 50)
    print(f"ML Success Rate (%): {100*success_rates[0]}")


def get_dft_data(targets):
    """
    Organizes the released target mapping for evaluation lookup.

    oc20dense_targets.pkl:
        ['system_id 1': [('config_id 1', dft_adsorption_energy), ('config_id 2', dft_adsorption_energy)], `system_id 2]

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


def get_dft_compute(counts):
    """
    Calculates the total DFT compute associated with establishing a ground
    truth using the released DFT timings: oc20dense_compute.pkl.

    Compute is measured in the total number of self-consistent steps (SC). The
    total number of ionic steps is also included for reference.
    """
    total_ionic_calls = 0
    total_scf_calls = 0
    for system in counts:
        for config in counts[system]:
            # Only count compute for adslab configurations. Clean surfaces are
            # excluded as they are used by both DFT and ML.
            if config != "surface":
                total_ionic_calls += counts[system][config]["ionic"]
                total_scf_calls += counts[system][config]["scf"]

    return total_ionic_calls, total_scf_calls


def filter_ml_data(ml_data, dft_data):
    """
    For ML systems in which no configurations made it through the physical
    constraint checks, set energies to an arbitrarily high value to ensure
    a failure case in evaluation.
    """
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
                    "scf_steps": 0,
                    "ionic_steps": 0,
                }
                ml_data[system][config] = _dict

    # for ML systems with no available ml+dft datapoints, set to an arbitrarily
    # high energy value
    for system in ml_data:
        for config in ml_data[system]:
            if not ml_data[system][config]["ml+dft_energy"]:
                ml_data[system][config]["ml+dft_energy"] = 1e10

    return ml_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file", required=True, help="Path to results-file to evaluate."
    )
    parser.add_argument(
        "--dft-targets",
        default="oc20dense_targets.pkl",
        help="Path to released target file: oc20dense_targets.pkl",
    )
    parser.add_argument(
        "--dft-compute",
        default="oc20dense_compute.pkl",
        help="Path to released compute file: oc20dense_compute.pkl",
    )
    parser.add_argument(
        "--maxk", default=5, type=int, help="Max top-k to retrieve results for."
    )
    parser.add_argument(
        "--ml-success",
        action="store_true",
        help="""
        Whether to compute an ML-only success rate. Here ML energies are
        directly use to compute a success rate if the predicted energy is considered
        valid. An ML prediction is considered valid if its energy is within 0.1
        eV of the DFT energy of the predicted configuration. This ensures the
        metric can't be games by predicting arbitrarily low values.'
        """,
    )
    args = parser.parse_args()

    targets = pickle.load(open(args.dft_targets, "rb"))
    gt_dft_counts = pickle.load(open(args.dft_compute, "rb"))

    ###### Process DFT Data ######
    dft_data = get_dft_data(targets)
    (total_ground_truth_ionic_calls, total_ground_truth_scf_calls) = get_dft_compute(
        gt_dft_counts
    )

    ###### Process ML Data ######
    ml_data = pickle.load(open(args.results_file, "rb"))
    ml_data = filter_ml_data(ml_data, dft_data)

    ###### Compute Metrics ######
    print(args.results_file)
    if not args.ml_success:
        "Compute AdsorbML success rates (hybrid ML+DFT)"
        compute_hybrid_success(ml_data, dft_data, k=range(1, args.maxk + 1))
    else:
        "Compute ML success rates (ML only)"
        compute_valid_ml_success(ml_data, dft_data)
