"""
This script plots the difference between the performance of models with and without
PhAST.
"""
print("Imports...", end="")
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms


# -----------------------
# -----  Constants  -----
# -----------------------

dict_metrics = {
    "names": {
        "tpr": "TPR, Recall, Sensitivity",
        "tnr": "TNR, Specificity, Selectivity",
        "fpr": "FPR",
        "fpt": "False positives relative to image size",
        "fnr": "FNR, Miss rate",
        "fnt": "False negatives relative to image size",
        "mpr": "May positive rate (MPR)",
        "mnr": "May negative rate (MNR)",
        "accuracy": "Accuracy (ignoring may)",
        "error": "Error",
        "f05": "F05 score",
        "precision": "Precision",
        "edge_coherence": "Edge coherence",
        "accuracy_must_may": "Accuracy (ignoring cannot)",
    },
    "key_metrics": ["mae_phast_impr", "time_phast_fraction"],
}

dict_archs = OrderedDict(
    [
        ("SchNet", "SchNet"),
        ("D++", "DimeNet++"),
        ("ForceNet", "ForceNet"),
    ]
)
dict_val = OrderedDict(
    [
        ("id", "ID"),
        ("ad", "OOD-ad"),
        ("cat", "OOD-cat"),
        ("both", "OOD-both"),
    ]
)
dict_models = OrderedDict(
    [
        ("baseline", "Baseline"),
        ("phast", "PhAST"),
    ]
)

# Model features
model_feats = [
    "masker",
    "seg",
    "depth",
    "dada_seg",
    "dada_masker",
    "spade",
    "pseudo",
    "ground",
    "instagan",
]

# Colors
crest = sns.color_palette("crest", as_cmap=False, n_colors=4)
palette_val = crest
sns.palplot(palette_val)
set2 = sns.color_palette("Set2", as_cmap=False, n_colors=3)
palette_baseline_phast = [set2[1], set2[2]]
sns.palplot(set2)
vlag = sns.color_palette("vlag", as_cmap=False, n_colors=3)
palette_baseline_phast = [vlag[2], vlag[0]]
sns.palplot(vlag)


# Markers
dict_mae_markers = OrderedDict([("id", "o"), ("ad", "s"), ("cat", "^"), ("both", "h")])
dict_time_markers = OrderedDict([("baseline", "d"), ("phast", "*")])


def parsed_args():
    """
    Parse and returns command-line args

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="results.csv",
        type=str,
        help="CSV containing the main results",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--dpi",
        default=150,
        type=int,
        help="DPI for the output images",
    )
    parser.add_argument(
        "--n_bs",
        default=1e6,
        type=int,
        help="Number of bootrstrap samples",
    )
    parser.add_argument(
        "--alpha",
        default=0.99,
        type=float,
        help="Confidence level",
    )
    parser.add_argument(
        "--bs_seed",
        default=17,
        type=int,
        help="Bootstrap random seed, for reproducibility",
    )

    return parser.parse_args()


def trim_mean_wrapper(a):
    return trim_mean(a, proportiontocut=0.2)


def find_model_pairs(technique, model_feats):
    model_pairs = []
    for mi in df.loc[df[technique]].model_feats.unique():
        for mj in df.model_feats.unique():
            if mj == mi:
                continue

            if df.loc[df.model_feats == mj, technique].unique()[0]:
                continue

            is_pair = True
            for f in model_feats:
                if f == technique:
                    continue
                elif (
                    df.loc[df.model_feats == mj, f].unique()[0]
                    != df.loc[df.model_feats == mi, f].unique()[0]
                ):
                    is_pair = False
                    break
                else:
                    pass
            if is_pair:
                model_pairs.append((mi, mj))
                break
    return model_pairs


if __name__ == "__main__":
    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------
    args = parsed_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))

    # Determine output dir
    if args.output_dir is None:
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)

    # Store args
    output_yml = output_dir / "phast_summary_plot.yml"
    with open(output_yml, "w") as f:
        yaml.dump(vars(args), f)

    # Read CSV
    df = pd.read_csv(args.input_csv, index_col=False)

    # Build data set
    df_mae = pd.DataFrame(
        columns=["architecture", "val", "mae_phast_impr"]
    )
    df_time = pd.DataFrame(
        columns=["architecture", "model", "time", "phast_fraction"]
    )
    for arch in df.architecture.unique():
        for val in df.val.unique():
            df_arch_val = df.loc[(df.architecture == arch) & (df.val == val)]
            # MAE
            mae_baseline = df_arch_val.loc[df_arch_val.baseline == True]["mae"].values
            mae_phast = df_arch_val.loc[df_arch_val.baseline == False]["mae"].values
            mae_phast_impr = -100.0 * ((mae_phast - mae_baseline) / mae_baseline)
            # Update df
            dfaux = pd.DataFrame.from_dict(
                {
                    "architecture": arch,
                    "val": val,
                    "mae_phast_impr": mae_phast_impr,
                }
            )
            df_mae = pd.concat([df_mae, dfaux], axis=0, ignore_index=True)
            # Time
            time_baseline = df_arch_val.loc[df_arch_val.baseline == True]["inf. time"].values
            time_phast = df_arch_val.loc[df_arch_val.baseline == False]["inf. time"].values
            time_phast_fraction = 100 * (time_phast / time_baseline)
            # Update df
            dfaux = pd.DataFrame.from_dict(
                {
                    "architecture": arch,
                    "model": "baseline",
                    "time": time_baseline,
                    "phast_fraction": 0.,
                }
            )
            df_time = pd.concat([df_time, dfaux], axis=0, ignore_index=True)
            dfaux = pd.DataFrame.from_dict(
                {
                    "architecture": arch,
                    "model": "phast",
                    "time": time_phast,
                    "phast_fraction": time_phast_fraction,
                }
            )
            df_time = pd.concat([df_time, dfaux], axis=0, ignore_index=True)

    ### Plot

    # Set up plot
    sns.reset_orig()
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "Utopia",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "ITC Bookman",
                "Bookman",
                "Times",
                "Palatino",
                "Charter",
                "serif" "Bitstream Vera Serif",
                "DejaVu Serif",
            ]
        }
    )

    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=True, dpi=args.dpi, figsize=(8, 2)
    )

    metrics = ["mae_phast_impr", ""]
#     dict_ci = {m: {} for m in metrics}

    # MAE PhAST improvement
    ax = sns.pointplot(
        ax=axes[0],
        data=df_mae,
        order=dict_archs.keys(),
        x="mae_phast_impr",
        y="architecture",
        hue="val",
        hue_order=[k for k in dict_mae_markers.keys()],
        markers=[v for v in dict_mae_markers.values()],
        dodge=0.5,
        palette=palette_val,
#             errwidth=1.5,
#             scale=0.6,
        join=False,
#             estimator=trim_mean_wrapper,
#             ci=int(args.alpha * 100),
#             n_boot=args.n_bs,
#             seed=args.bs_seed,
    )
    # Legend
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    leg_labels = [dict_val[val] for val in leg_labels]
    leg = ax.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="center",
        title="",
        bbox_to_anchor=(-0.2, 1.1, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=-0.4,
        ncol=len(dict_val),
    )
    # Plot Baseline
    df_mae_baseline = pd.DataFrame(
            {"architecture": df.architecture.unique(), "mae": [0.0, 0.0, 0.0]}
    )
    ax = sns.pointplot(
        ax=axes[0],
        data=df_mae_baseline,
        order=dict_archs.keys(),
        x="mae",
        y="architecture",
        markers=dict_time_markers["baseline"],
        color=palette_baseline_phast[0],
#             errwidth=1.5,
#             scale=0.6,
        join=False,
#             estimator=trim_mean_wrapper,
#             ci=int(args.alpha * 100),
#             n_boot=args.n_bs,
#             seed=args.bs_seed,
    )
    # Set X-label
    ax.set_xlabel("MAE improvement [%]")

    # Time PhAST improvement
    ax = sns.pointplot(
        ax=axes[1],
        data=df_time,
        order=dict_archs.keys(),
        x="time",
        y="architecture",
        hue="model",
        hue_order=["baseline", "phast"],
        markers=[v for v in dict_time_markers.values()],
        palette=palette_baseline_phast,
#             errwidth=1.5,
#             scale=0.6,
        join=False,
#             estimator=trim_mean_wrapper,
#             ci=int(args.alpha * 100),
#             n_boot=args.n_bs,
#             seed=args.bs_seed,
    )
    # Legend
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    leg_labels = [dict_models[model] for model in leg_labels]
    leg = ax.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="center",
        title="",
        bbox_to_anchor=(-0.2, 1.1, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=-0.4,
        ncol=len(dict_val),
    )
    # Set X-label
    ax.set_xlabel("Inference time [s]")
    # X-lim
    ax.set_xlim(left=0.0, right=ax.get_xlim()[1])

    for ax in axes:

        # Change spines
        sns.despine(left=True, bottom=True)

        # Set Y-label
        ax.set_ylabel(None)

        # Y-tick labels
        ax.set_yticklabels(list(dict_archs.values()), fontsize="medium")

        # X-ticks
        xticks = ax.get_xticks()
        xticklabels = xticks
        if 0.0 not in xticks:
            xticks = np.append(xticks, 0.0)
            xticklabels = np.append(xticklabels, 0.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize="small")

        # Y-lim
        display2data = ax.transData.inverted()
        ax2display = ax.transAxes
        _, y_bottom = display2data.transform(ax.transAxes.transform((0.0, 0.02)))
        _, y_top = display2data.transform(ax.transAxes.transform((0.0, 0.98)))
        ax.set_ylim(bottom=y_bottom, top=y_top)

        # Draw line at H0
        if ax == axes[0]:
            y = np.arange(ax.get_ylim()[1], ax.get_ylim()[0], 0.1)
            x = 0.0 * np.ones(y.shape[0])
            ax.plot(x, y, linestyle=":", linewidth=2.0, color="black", zorder=1)

        # Draw gray area
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        width = np.abs(xlim[1] - xlim[0])
        offset = 0.05
        x0 = xlim[0] - offset * width
        width = width + 2 * offset * width
        ax.set_xlim(left=x0, right=x0 + width)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        height = 0.3
        margin = 0.05
        for idx in range(len(df_mae.architecture.unique())):
            rect = mpatches.Rectangle(
                xy=(x0, height * idx + margin * idx),
                width=width,
                height=height,
                transform=trans,
                linewidth=0.0,
                edgecolor="none",
                facecolor="gray",
                alpha=0.05,
            )
            ax.add_patch(rect)


    # Save figure
    output_fig = output_dir / "summary.png"
    fig.savefig(output_fig, dpi=fig.dpi, bbox_inches="tight")
