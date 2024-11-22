"""The script used to perform inference for HER"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairchem.applications.ocx.core.data_handling import load_and_preprocess_data
from fairchem.applications.ocx.core.features import add_el_features
from fairchem.applications.ocx.core.voltage import get_she
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

plt.rc("text", usetex=False)
plt.rc("font", family="serif", size=10)
plt.rc("text.latex", preamble=r"\usepackage{times}")
COLORS = ["#0064E0", "#C80A28"]  # Meta's blue and red


def get_computational_df_to_predict(computational_file: str):
    """
    Get aggregate energies per bulk to perform inference on materials
    in the database.

    Args:
    computational_file (str): path to the computational file

    """
    # Load data
    if computational_file.split(".")[-1] == "csv":
        df = pd.read_csv(computational_file, low_memory=False)
    elif computational_file.split(".")[-1] == "pkl":
        df = pd.read_pickle(computational_file)
    else:
        raise NotImplementedError(
            "please provide a pkl or csv for `computational_file`."
        )
    energies = [
        "CO_min_sp_e",
        "C_min_sp_e",
        "H_min_sp_e",
        "CHO_min_sp_e",
        "COCOH_min_sp_e",
        "OH_min_sp_e",
    ]
    df.dropna(subset=energies, inplace=True)
    dfg = (
        df.groupby(by="bulk_id")
        .agg(
            {
                "CO_min_sp_e": np.mean,
                "C_min_sp_e": np.mean,
                "H_min_sp_e": np.mean,
                "CHO_min_sp_e": np.mean,
                "COCOH_min_sp_e": np.mean,
                "OH_min_sp_e": np.mean,
                "slab_comp": "first",
            }
        )
        .reset_index()
    )
    dfg.rename(
        columns={
            "CO_min_sp_e": "CO_mean_energy",
            "C_min_sp_e": "C_mean_energy",
            "H_min_sp_e": "H_mean_energy",
            "CHO_min_sp_e": "CHO_mean_energy",
            "COCOH_min_sp_e": "COCOH_mean_energy",
            "OH_min_sp_e": "OH_mean_energy",
        },
        inplace=True,
    )
    return dfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expt_csv",
        help="path to the experimental csv. Available in ../data/experimental_data",
    )
    parser.add_argument(
        "--comp_csv",
        help="path to the computational csv. Available for download, see README",
    )
    parser.add_argument(
        "--output_dir", help="path to the output directory where files will be saved"
    )
    parser.add_argument(
        "--xrd_csv", help="path to the XRD csv. Available in ../data/experimental_data"
    )
    parser.add_argument(
        "--max_rwp",
        default=40,
        type=float,
        help="maximum Rwp value (from Rietveld refinement) to filter XRD data",
    )
    parser.add_argument(
        "--min_q_score",
        default=70,
        type=float,
        help="minimum q score to filter XRD data. For q score definition see the manuscript",
    )
    parser.add_argument(
        "--cod_lookup",
        default=None,
        help="path to the COD lookup file which matches COD structures to those contains in the computational workflow",
    )
    parser.add_argument(
        "--source",
        default="both",
        help="the experimental synthesis source. Options are `both`, `vsp`, and `uoft`",
    )
    parser.add_argument(
        "--no_energy",
        action="store_true",
        default=False,
        help="If energies should be used as features",
    )
    parser.add_argument(
        "--add_features",
        action="store_true",
        default=False,
        help="If matminer elemental features should be added",
    )
    parser.add_argument(
        "--sigma",
        default=0.0425858308961896,
        type=float,
        help="Used for plotting the sigma values in the parity plot.",
    )
    parser.add_argument(
        "--cv_results_csv",
        default=None,
        help="Path to the results csv for training with cross validation (from ml_train.py).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    feature_strings = []

    # process and load data
    df_comp_pred = get_computational_df_to_predict(args.comp_csv)

    df, df_expt_const_v, df_raw_expt = load_and_preprocess_data(
        args.comp_csv,
        args.expt_csv,
        xrd_data_file=args.xrd_csv,
        max_rwp=args.max_rwp,
        min_q_score=args.min_q_score,
        cod_to_ocp_lookup=args.cod_lookup,
        reaction="HER",
    )

    df_expt = df_expt_const_v

    db_metadata = ["composition", "source", "batch number", "matched"]
    adsorbate_descriptors = ["H", "OH"]

    energy_descriptors = []
    for ads in adsorbate_descriptors:
        energy_descriptors.append(f"{ads}_mean_energy")

    if not args.no_energy:
        features = energy_descriptors
    else:
        features = []

    if args.add_features:
        df_expt = add_el_features(
            df_expt,
            composition_column="xrf comp",
        )
        df_comp_pred = add_el_features(
            df_comp_pred,
            composition_column="slab_comp",
        )
        for ft in [
            "X",
            "row",
            "group",
            "atomic_mass",
            "atomic_radius",
            "mendeleev_no",
        ]:
            features.append(f"PymatgenData mean {ft}")

    print(f"There are {len(features)} features: {features}")

    products = ["voltage"]

    df_expt.dropna(subset=features, inplace=True)
    _df = df_expt.copy()
    x = _df[features]
    x = pd.DataFrame(x)

    for col in x:
        mean_ = x[col].mean()
        std_ = x[col].max()
        x[col] = (x[col] - mean_) / std_
        df_comp_pred[col] = (df_comp_pred[col] - mean_) / std_
        df_expt[col] = (df_expt[col] - mean_) / std_

    inputs = np.stack([df_comp_pred[col].tolist() for col in x.columns], axis=-1)
    x = x.values
    for product in products:
        y = _df[product].values
        model = LinearRegression()

        # Normalize input features
        model.fit(x, y)

        predictions = model.predict(inputs)
        trained_preds = model.predict(x)
        df_expt[f"predicted_{product}"] = trained_preds
        df_comp_pred[f"predicted_{product}"] = predictions

    df_comp_pred.to_pickle(os.path.join(args.output_dir, "full_comp_inference.pkl"))
    df_expt.to_pickle(os.path.join(args.output_dir, "trained_inference.pkl"))

    # MAKE VOLCANO FIGURE
    os.chdir(args.output_dir)
    # Perform PCA on the mean energies
    model = PCA(n_components=1)
    x = model.fit_transform(df_comp_pred[features])
    ex_var = model.explained_variance_ratio_[0]

    print(
        f"{ex_var*100:1.0f} % of the variance is explained by the first principal component"
    )

    train_x = model.transform(df_expt[features])
    df_comp_pred["pc1"] = x
    df_expt["pc1"] = train_x

    # get the predicted voltage as V v. SHE
    df_comp_pred["predicted_voltage_she"] = df_comp_pred.predicted_voltage.apply(
        get_she
    )
    df_expt["voltage_she"] = df_expt.voltage.apply(get_she)
    df_expt["predicted_voltage_she"] = df_expt.predicted_voltage.apply(get_she)

    # plot the volcano plot
    f = sns.displot(
        df_comp_pred,
        x="pc1",
        y="predicted_voltage_she",
        kind="kde",
        color=COLORS[0],
    )
    sns.scatterplot(df_expt, x="pc1", y="predicted_voltage_she", color="#465A69")
    sns.scatterplot(
        df_comp_pred[df_comp_pred.bulk_id == "mp-126"],
        x="pc1",
        y="predicted_voltage_she",
        marker="*",
        color="#C80A28",
        s=400,
    )
    if args.add_features:
        f.set(
            xlabel="PC1",
            ylabel="Predicted Voltage v. SHE",
            xlim=(-1, 1),
            ylim=(-2.5, 0),
        )
    else:
        f.set(
            xlabel="PC1",
            ylabel="Predicted Voltage v. SHE",
            xlim=(-2.5, 2.5),
            ylim=(-1.8, -0.8),
        )
    f.savefig("pc1_v_voltage.svg")

    # MAKE PARITY PLOT
    if args.cv_results_csv is not None:
        df_expt = pd.read_csv(args.cv_results_csv)
        df_expt.dropna(subset=["voltage_she_expt", "voltage_she_pred"], inplace=True)
        f, ax = plt.subplots(1, 1)
        x1 = df_expt[df_expt.matched > 0]["voltage_she_expt"].tolist()
        y1 = df_expt[df_expt.matched > 0]["voltage_she_pred"].tolist()

        x2 = df_expt[df_expt.matched == 0]["voltage_she_expt"].tolist()
        y2 = df_expt[df_expt.matched == 0]["voltage_she_pred"].tolist()

        ax.scatter(x1, y1, color=COLORS[0], label="Matched")
        ax.scatter(x2, y2, edgecolor=COLORS[0], facecolor="none", label="Unmatched")
        slope, intercept, r, p, se = linregress(x1 + x2, y1 + y2)
        min_ = min(min(x1 + x2), min(y1 + y2))
        max_ = max(max(x1 + x2), max(y1 + y2))
        parity_x = np.array([min_, max_])
        parity_y = np.array(
            [
                min_ * slope + intercept,
                max_ * slope + intercept,
            ]
        )
        ax.plot(
            parity_x,
            parity_y,
            "--",
            linewidth=2,
            color="k",
        )
        ax.fill_between(
            parity_x,
            parity_y - args.sigma * 2,
            parity_y + args.sigma * 2,
            color="gray",
            alpha=0.2,
        )
        mae = mean_absolute_error(np.array(y1 + y2), np.array(x1 + x2))
        ax.set_xlim([min_, max_])
        ax.set_ylim([min_, max_])
        ax.set_xlabel("Experimental Voltage v. SHE")
        ax.set_ylabel("Predicted Voltage v. SHE")
        plt.grid(linestyle="dotted")
        ax.annotate(f"R2: {r**2:.2f}", (0.05, 0.95), xycoords="axes fraction")
        ax.annotate(f"MAE: {mae:.2f} V v. SHE", (0.05, 0.9), xycoords="axes fraction")
        ax.set_aspect("equal")
        ax.legend()
        f.savefig("parity_plot.svg")
