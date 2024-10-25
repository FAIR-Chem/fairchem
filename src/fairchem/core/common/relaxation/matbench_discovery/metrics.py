"""Evaluate matbench-discovery benchmark

Adapted from https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/metrics.py
using our lmdbs to evaluate, see wbm2aselmdb.py for info on WBM lmdbs

To run an evaluation WBM structures must be relaxed and the relaxed energies should be passed to the evaluation

Target energies above hull and formation energies are saved in the following data dictionaries in ASE DBs
/private/home/lbluque/large_experiments/opencatalyst/foundation_models/evaluation/matbench-discovery/WBM_IS2RE.aselmdb
/private/home/lbluque/large_experiments/opencatalyst/foundation_models/evaluation/matbench-discovery/WBM_IS2RE_unique_protos.aselmdb

energy_above_hull_target-> e_above_hull_mp2020_corrected_ppd_mp
formation_energy_target-> e_form_per_atom_mp2020_corrected
"""

import torch
from .formation_energy import get_formation_energy_per_atom


def classify_stable_from_predicted_energy(
    relaxed_energy_predictions: torch.Tensor,
    energy_above_hull_target: torch.Tensor,
    formation_energy_target: torch.Tensor,
    atomic_numbers: tuple[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate matbench-discovery metrics

    Note if predicting MP corrected energies all targets and predictions must be corrected.
    If not predicting corrected energies, then nothing should have corrections.
    Matbench-discovery defaults to corrections, but valid metrics can be computed without them.

    Args:
        relaxed_energy_predictions: predicted total relaxed energies
        energy_above_hull_target: target energies above hull per atom
        formation_energy_target: target formation energies per atom
        atomic_numbers: tuple of tensors with atomic numbers for each predicted energy
    Returns:
        tuple[TP, FN, FP, TN]: Indices as torch.Tensor for true positives,
            false negatives, false positives and true negatives (in this order).
    """
    formation_energy_predictions = get_formation_energy_per_atom(relaxed_energy_predictions, atomic_numbers)
    energy_above_hull_predictions = energy_above_hull_target - formation_energy_target + formation_energy_predictions
    return classify_stable(energy_above_hull_target, energy_above_hull_predictions)


def calculate_stability_metrics(
    true_pos: torch.Tensor,
    false_neg: torch.Tensor,
    false_pos: torch.Tensor,
    true_neg: torch.Tensor
) -> dict[str, float]:
    """Get a dictionary of stability prediction metrics. Mostly binary classification
    metrics, but also MAE, RMSE and R2.

    Returns:
        dict[str, float]: dictionary of classification metrics with keys DAF, Precision,
            Recall, Accuracy, F1, TPR, FPR, TNR, FNR, MAE, RMSE, R2.
    """
    n_preds = len(true_pos)
    n_true_pos, n_false_neg, n_false_pos, n_true_neg = map(
        sum,
        (true_pos, false_neg, false_pos, true_neg)
    )

    n_total_pos = n_true_pos + n_false_neg
    n_total_neg = n_true_neg + n_false_pos

    # prevalence: dummy discovery rate of stable crystals by selecting randomly from
    # all materials
    prevalence = n_total_pos / (n_total_pos + n_total_neg)
    precision = n_true_pos / (n_true_pos + n_false_pos)  # model's discovery rate
    recall = n_true_pos / n_total_pos

    TPR = recall
    FPR = n_false_pos / n_total_neg
    TNR = n_true_neg / n_total_neg
    FNR = n_false_neg / n_total_pos

    if FPR + TNR != 1:  # sanity check: false positives + true negatives = all negatives
        raise ValueError(f"{FPR=} {TNR=} don't add up to 1")

    if TPR + FNR != 1:  # sanity check: true positives + false negatives = all positives
        raise ValueError(f"{TPR=} {FNR=} don't add up to 1")

    return dict(
        F1=2 * (precision * recall) / (precision + recall),
        DAF=precision / prevalence,
        Precision=precision,
        Recall=recall,
        Accuracy=(n_true_pos + n_true_neg) / n_preds,
        **dict(TPR=TPR, FPR=FPR, TNR=TNR, FNR=FNR),
        **dict(TP=n_true_pos, FP=n_false_pos, TN=n_true_neg, FN=n_false_neg),
    )


def classify_stable(
    e_above_hull_pred: torch.Tensor,
    e_above_hull_true: torch.Tensor,
    *,
    stability_threshold: float | None = 0,
    fillna: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify model stability predictions as true/false positive/negatives (usually
    w.r.t DFT-ground truth labels). All energies are assumed to be in eV/atom
    (but shouldn't really matter as long as they're consistent).

    Args:
        e_above_hull_true (pd.Series): Ground truth energy above convex hull values.
        e_above_hull_pred (pd.Series): Model predicted energy above convex hull values.
        stability_threshold (float | None, optional): Maximum energy above convex hull
            for a material to still be considered stable. Usually 0, 0.05 or 0.1.
            Defaults to 0, meaning a material has to be directly on the hull to be
            called stable. Negative values mean a material has to pull the known hull
            down by that amount to count as stable. Few materials lie below the known
            hull, so only negative values very close to 0 make sense.
        fillna (bool): Whether to fill NaNs as the model predicting unstable. Defaults
            to True.

    Returns:
        tuple[TP, FN, FP, TN]: Indices as torch.Tensor for true positives,
            false negatives, false positives and true negatives (in this order).
    """
    actual_pos = e_above_hull_true <= (stability_threshold or 0)  # guard against None
    actual_neg = e_above_hull_true > (stability_threshold or 0)

    model_pos = e_above_hull_pred <= (stability_threshold or 0)
    model_neg = e_above_hull_pred > (stability_threshold or 0)

    if fillna:
        nan_mask = torch.isnan(e_above_hull_pred)
        model_pos[nan_mask] = False  # fill NaNs as unstable
        model_neg[nan_mask] = True  # fill NaNs as unstable

        n_pos, n_neg, total = model_pos.sum(), model_neg.sum(), len(e_above_hull_pred)
        if n_pos + n_neg != total:
            raise ValueError(
                f"after filling NaNs, the sum of positive ({n_pos}) and negative "
                f"({n_neg}) predictions should add up to {total=}"
            )

    true_pos = actual_pos & model_pos
    false_neg = actual_pos & model_neg
    false_pos = actual_neg & model_pos
    true_neg = actual_neg & model_neg

    return true_pos, false_neg, false_pos, true_neg
