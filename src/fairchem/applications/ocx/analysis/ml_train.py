"""
Driver script to evaluate the performance of ML models on predicting
experimental products for HER and CO2RR in the OCx24 dataset.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
from fairchem.applications.ocx.core.data_handling import load_and_preprocess_data
from fairchem.applications.ocx.core.features import add_el_features
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)

MODEL_MAP = {
    "linear": LinearRegression(),
    "rf": RandomForestRegressor(n_jobs=10, random_state=42),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Always define these
    parser.add_argument(
        "--source",
        choices=["both", "vsp", "uoft"],
        default="both",
        help="Experimental source to run analysis on.",
    )
    parser.add_argument(
        "--reaction",
        choices=["HER", "CO2R"],
        default="CO2R",
        help="Electrochemical reaction to run analysis on.",
    )
    parser.add_argument(
        "--add_el_features",
        action="store_true",
        default=False,
        help="Include Matminer features.",
    )
    parser.add_argument(
        "--energy_aggregation",
        choices=["mean", "wulff", "boltz"],
        help="How to aggregate adsorption energy information",
    )
    parser.add_argument(
        "--filter_on_matched",
        action="store_true",
        default=False,
        help="Only consider XRD matched samples",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save figures + results"
    )

    # If processing from scratch, these must also be defined
    parser.add_argument("--expt_csv", type=str, help="Path to experimental csv file.")
    parser.add_argument("--comp_csv", type=str, help="Path to computational csv file.")
    parser.add_argument("--xrd_csv", type=str, help="Path to xrd csv file")
    parser.add_argument("--cod_lookup", type=str, help="Path to cod mapping file")
    parser.add_argument(
        "--max_rwp", type=int, default=40, help="Maximum RWP value to consider"
    )
    parser.add_argument(
        "--min_q_score", type=int, default=70, help="Minimum Q score to consider"
    )
    parser.add_argument(
        "--voltage", type=float, default=3.3, help="Voltage to interpolate for CO2R"
    )

    # If processing from a processed dataframe, this must be defined otherwise it is optional and processed dataframe will be saved at this location
    parser.add_argument(
        "--processed_df",
        type=str,
        default=None,
        help="Path to processed df. If not available, specify other arguments and processed df will be saved at this location",
    )

    args = parser.parse_args()

    if type(args.processed_df) == str and os.path.isfile(args.processed_df):
        df_expt = pd.read_csv(args.processed_df)
        logging.info(f"Loading processed data at {args.processed_df}")
    else:
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
        if args.processed_df:
            df_expt.to_csv(args.processed_df)
            logging.info(f"Processed data saved to {args.processed_df}")

    if args.source in ["vsp", "uoft"]:
        df_expt = df_expt[df_expt.source == args.source]
    # add elemental features
    if args.add_el_features:
        df_expt = add_el_features(
            df_expt,
            composition_column="xrf comp",
        )
    if args.filter_on_matched:
        df_expt = df_expt[df_expt.matched]

    db_metadata = ["composition", "source", "batch number", "matched"]
    if args.reaction == "CO2R":
        adsorbate_descriptors = ["C", "CO", "CHO", "COCOH", "H", "OH"]
    elif args.reaction == "HER":
        adsorbate_descriptors = ["H", "OH"]

    energy_descriptors = []
    for ads in adsorbate_descriptors:
        if args.energy_aggregation == "mean":
            energy_descriptors.append(f"{ads}_mean_energy")
        elif args.energy_aggregation == "wulff":
            energy_descriptors.append(f"{ads}_wulff_energy")
        elif args.energy_aggregation == "boltz":
            energy_descriptors.append(f"{ads}_boltz_energy")

    features = energy_descriptors
    if args.add_el_features:
        for ft in [
            "X",
            "row",
            "group",
            "atomic_mass",
            "atomic_radius",
            "mendeleev_no",
        ]:
            features.append(f"PymatgenData mean {ft}")

    if args.reaction == "CO2R":
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
        products = ["voltage_she"]

    for model_name in MODEL_MAP:
        for cv_type in ["loco_cv", "loo_cv"]:
            for product in products:
                _df = df_expt.copy()
                x = _df[features].values
                y = _df[product].values
                splits = _df[cv_type].values
                unique_splits = np.unique(splits)

                CVSplits = []
                for split in unique_splits:
                    train_indices = (splits != split).nonzero()[0]
                    test_indices = (splits == split).nonzero()[0]
                    CVSplits.append((train_indices, test_indices))
                cv = CVSplits

                model = MODEL_MAP[model_name]
                # Standardize input features
                pipeline = Pipeline(
                    [("scale_x", preprocessing.StandardScaler()), ("model", model)]
                )
                predictions = model_selection.cross_val_predict(
                    pipeline, x, y, cv=cv, verbose=2
                )

                # Build results dataframe
                pd_columns = [*db_metadata]
                predictions_db = pd.DataFrame(columns=pd_columns)
                predictions_db[db_metadata] = _df[db_metadata].values

                pred = predictions.ravel()
                target = y.ravel()

                predictions_db[f"{product}_pred"] = pred
                predictions_db[f"{product}_expt"] = target
                predictions_db[f"{product}_error"] = abs(pred - target)

                fig = px.scatter(
                    predictions_db,
                    x=f"{product}_expt",
                    y=f"{product}_pred",
                    hover_data=[
                        "composition",
                        "source",
                        "batch number",
                        f"{product}_error",
                    ],
                    trendline="ols",
                    trendline_scope="overall",
                    color="source",
                )

                # Annotate plot
                r2 = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
                mae = mean_absolute_error(
                    predictions_db[f"{product}_expt"],
                    predictions_db[f"{product}_pred"],
                )

                fig.add_annotation(
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"R2: {np.round(r2, 3)}",
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.add_annotation(
                    x=1,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"MAE: {mae}",
                    showarrow=False,
                    font=dict(size=16),
                )

                if args.reaction == "HER":
                    xaxis = "Experimental [V vs SHE]"
                    yaxis = "ML Predicted [V vs SHE]"
                elif args.reaction == "CO2R" and "fe" in product:
                    xaxis = "Experimental [%]"
                    yaxis = "ML Predicted [%]"
                elif args.reaction == "CO2R" and "pr" in product:
                    xaxis = "Experimental [mol/cm^2 s]"
                    yaxis = "ML Predicted [mol/cm^2 s]"

                feature_string = (
                    f"energy_{args.energy_aggregation}_matminer_{args.add_el_features}"
                )
                title = f"source-{args.source.lower()} | {cv_type} | {product.lower()} | {feature_string}"
                fig.update_layout(
                    title=title,
                    xaxis_title=xaxis,
                    yaxis_title=yaxis,
                    font=dict(size=18),
                    width=1200,
                    height=900,
                )
                outputdir = os.path.join(
                    args.output_dir,
                    args.reaction,
                    f"filter{args.filter_on_matched}",
                    cv_type,
                    model_name,
                    product,
                    args.source + "_" + feature_string,
                )
                os.makedirs(outputdir, exist_ok=True)
                fig.write_html(os.path.join(outputdir, "figure.html"))
                predictions_db.to_csv(os.path.join(outputdir, "results.csv"))
