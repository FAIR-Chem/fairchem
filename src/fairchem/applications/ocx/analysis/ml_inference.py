from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from automated_analysis.core.data_handling import load_and_preprocess_data
from automated_analysis.core.features import add_el_features
from automated_analysis.core.voltage import get_she
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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
    parser.add_argument("--expt_csv")
    parser.add_argument("--comp_csv")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--average_over_replicates_fixed_current_density",
        action="store_true",
        default=False,
    )
    parser.add_argument("--voltage", default=3.3, type=float)
    parser.add_argument("--xrd_csv", default=None)
    parser.add_argument("--max_rwp", default=None, type=float)
    parser.add_argument("--min_q_score", default=None, type=float)
    parser.add_argument("--cod_lookup", default=None)
    parser.add_argument("--source", default="both")
    parser.add_argument("--reaction", default="CO2R")
    parser.add_argument("--add_features", choices=["matminer", "comp"], default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    feature_strings = []

    # process and load data
    df_comp_pred = get_computational_df_to_predict(args.comp_csv)

    comp_df, df_expt, df_raw_expt = load_and_preprocess_data(
        computational_file=args.comp_csv,
        experimental_file=args.expt_csv,
        xrd_data_file=args.xrd_csv,
        max_rwp=args.max_rwp,
        min_q_score=args.min_q_score,
        cod_to_ocp_lookup=args.cod_lookup,
        interpolate_v=args.voltage,
        reaction=args.reaction,
        filter_on_matched=False,
    )

    # add elemental features from matminer, magpie, etc.
    if args.add_features:
        original_columns = df_expt.columns
        df_expt = add_el_features(
            df_expt,
            feature_type=args.add_features,
            composition_column="xrf comp",
        )

        df_comp_pred = add_el_features(
            df_comp_pred,
            eature_type=args.add_features,
            composition_column="slab_comp",
        )
        addl_features = list(set(df_expt.columns) - set(original_columns))

        feature_strings.append(args.add_features)

    db_metadata = ["composition", "source", "batch number", "matched"]
    if args.reaction == "CO2R":
        adsorbate_descriptors = ["C", "CO", "CHO", "COCOH", "H", "OH"]
    elif args.reaction == "HER":
        adsorbate_descriptors = ["H", "OH"]

    energy_descriptors = []
    for ads in adsorbate_descriptors:
        energy_descriptors.append(f"{ads}_mean_energy")

    features = energy_descriptors

    if args.add_features == "matminer":
        for ft in [
            "X",
            "row",
            "group",
            # "block",
            "atomic_mass",
            "atomic_radius",
            "mendeleev_no",
        ]:
            features.append(f"PymatgenData mean {ft}")
    if args.add_features == "comp":
        features += addl_features

    print(f"There are {len(features)} features: {features}")

    if args.reaction == "CO2R":
        # predict single products independently
        # also try predicting all outputs simultaneously
        products = [
            "fe_co",
            "fe_h2",
            "fe_ch4",
            "fe_c2h4",
            "fe_liquid",
            "H2_pr",
            "CO_pr",
            "CH4_pr",
            "Total_Liquid_pr",
            "C2H4_pr",
        ]
    elif args.reaction == "HER":
        products = ["voltage"]

    df_expt.dropna(subset=features, inplace=True)
    _df = df_expt.copy()
    x = _df[features]
    x = pd.DataFrame(x)

    for col in x:
        min_ = x[col].min()
        max_ = x[col].max()
        x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())
        df_comp_pred[col] = (df_comp_pred[col] - min_) / (max_ - min_)
        df_expt[col] = (df_expt[col] - min_) / (max_ - min_)

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

    if args.reaction == "HER":
        # MAKE VOLCANO FIGURE
        # Perform PCA on the mean energies
        model = PCA(n_components=1)
        x = model.fit_transform(df_comp_pred[["H_mean_energy", "OH_mean_energy"]])
        ex_var = model.explained_variance_ratio_

        print(
            f"{ex_var*100:1.0f} % of the variance is explained by the first principal component"
        )

        train_x = model.transform(df_expt[["H_mean_energy", "OH_mean_energy"]])
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
        f.set(
            xlabel="PC1",
            ylabel="Predicted Voltage v. SHE",
            xlim=(-1.5, 1.5),
            ylim=(-1.8, -0.8),
        )
        f.savefig("pc1_v_voltage.svg")
