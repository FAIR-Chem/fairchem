"""
This script plots the difference between the performance of models with and without
PhAST.
"""
import sys
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from utils import plot_setup
from utils import get_palette_val, get_palette_methods


def preprocess_df(df, config):
    # Seed
    df["seed"] = [None for _ in range(len(df))]
    for seed in config.seeds:
        df.loc[df["Notes"].str.contains(f"s{seed}", regex=False), "seed"] = seed
    # Architecture
    df["architecture"] = [None for _ in range(len(df))]
    for architecture in config.architectures:
        df.loc[
            df["Name"].str.contains(architecture.key, regex=False), "architecture"
        ] = architecture.name
    # Baseline
    df["baseline"] = [False for _ in range(len(df))]
    df.loc[df["Notes"].str.contains("Baseline", regex=False), "baseline"] = True
    # Rename validation split columns
    rename_dict = {split.key: split.short for split in config.val_splits}
    df = df.rename(columns=rename_dict)
    # Melt validation split columns
    value_vars = [v for v in rename_dict.values()]
    df = pd.melt(
        df,
        id_vars=["architecture", "baseline", "seed"],
        value_vars=value_vars,
        var_name="val",
        value_name="mae",
    )
    return df


def make_plot_dataframes(df_orig, df_orig_times):
    # Build data set
    df_mae = pd.DataFrame(columns=["architecture", "val", "mae_phast_impr", "seed"])
    df_time = pd.DataFrame(columns=["architecture", "model", "time", "phast_fraction"])
    for val in df_orig.val.unique():
        df_val = df_orig.loc[(df_orig.val == val)]
        for arch in df_val.architecture.unique():
            df_arch = df_val.loc[(df_val.architecture == arch)]
            time_phast = df_orig_times.loc[
                (df_orig_times.val == val)
                & (df_orig_times.architecture == arch)
                & (df_orig_times.baseline == False)
            ]["inf. time"].values
            time_baseline = df_orig_times.loc[
                (df_orig_times.val == val)
                & (df_orig_times.architecture == arch)
                & (df_orig_times.baseline == True)
            ]["inf. time"].values
            for seed_phast in df_arch.seed.unique():
                mae_phast = df_arch.loc[
                    (df_arch.seed == seed_phast) & (df_arch.baseline == False)
                ]["mae"].values
                for seed_baseline in df_arch.seed.unique():
                    mae_baseline = df_arch.loc[
                        (df_arch.seed == seed_baseline) & (df_arch.baseline == True)
                    ]["mae"].values
                    # MAE
                    mae_phast_impr = -100.0 * (
                        (mae_phast - mae_baseline) / mae_baseline
                    )
                    # Update df
                    dfaux = pd.DataFrame.from_dict(
                        {
                            "architecture": arch,
                            "val": val,
                            "mae_phast_impr": mae_phast_impr,
                            "seed": seed_phast,
                        }
                    )
                    df_mae = pd.concat([df_mae, dfaux], axis=0, ignore_index=True)
                    # Time
                    time_phast_fraction = 100 * (time_phast / time_baseline)
                    # Update df
                    dfaux = pd.DataFrame.from_dict(
                        {
                            "architecture": arch,
                            "model": "baseline",
                            "time": time_baseline,
                            "phast_fraction": 0.0,
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

    return df_mae, df_time


def plot(df_orig, df_mae, df_time, config):
    plot_setup()

    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=True, dpi=config.plot.dpi, figsize=(8, 2)
    )

    # MAE PhAST improvement
    ax = sns.pointplot(
        ax=axes[0],
        data=df_mae,
        estimator=np.mean,
        errorbar="ci",
        n_boot=config.plot.bs_n,
        seed=config.plot.bs_seed,
        order=[el.name for el in config.data.architectures],
        x="mae_phast_impr",
        y="architecture",
        hue="val",
        hue_order=[el.short for el in config.data.val_splits],
        markers=[
            config.plot.markers.val_splits[el.short] for el in config.data.val_splits
        ],
        dodge=config.plot.dodge,
        palette=get_palette_val(
            config.plot.colors.val.palette, len(config.data.val_splits)
        ),
        errwidth=config.plot.errwidth,
        scale=0.0,
        join=False,
    )
    for arch in df_orig.architecture.unique():
        ax = sns.pointplot(
            ax=axes[0],
            data=df_mae.loc[df_mae["architecture"] == arch],
            estimator=np.mean,
            errorbar=None,
            order=[el.name for el in config.data.architectures],
            x="mae_phast_impr",
            y="architecture",
            hue="val",
            hue_order=[el.short for el in config.data.val_splits],
            markers=config.plot.markers.architectures[arch],
            dodge=config.plot.dodge,
            palette=get_palette_val(
                config.plot.colors.val.palette, len(config.data.val_splits)
            ),
            errwidth=0.0,
            scale=config.plot.scale,
            join=False,
        )
    # Legend val
    leg_handles = []
    for val_split, color in zip(
        config.data.val_splits,
        get_palette_val(config.plot.colors.val.palette, len(config.data.val_splits)),
    ):
        leg_handles.append(mpatches.Patch(color=color, label=val_split.name))
    leg2 = ax.legend(
        handles=leg_handles,
        loc="center",
        title="",
        bbox_to_anchor=(-0.2, 1.09, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=0.4,
        ncol=len(config.data.val_splits),
    )
    if config.plot.plot_baseline:
        # Plot Baseline
        df_mae_baseline = pd.DataFrame(
            {"architecture": df_orig.architecture.unique(), "mae": [0.0, 0.0, 0.0]}
        )
        ax = sns.pointplot(
            ax=axes[0],
            data=df_mae_baseline,
            order=[el.name for el in config.data.architectures],
            x="mae",
            y="architecture",
            markers=config.plot.markers.models["baseline"],
            color=get_palette_methods(config.plot.colors.methods.palette)[0],
            join=False,
        )
    # Set X-label
    ax.set_xlabel("MAE improvement [%]")

    # Time PhAST improvement
    ax = sns.pointplot(
        ax=axes[1],
        data=df_time,
        order=[el.name for el in config.data.architectures],
        x="time",
        y="architecture",
        hue="model",
        hue_order=[el.key for el in config.data.models],
        markers=[config.plot.markers.models[el.key] for el in config.data.models],
        palette=get_palette_methods(config.plot.colors.methods.palette),
        join=False,
    )
    # Legend
    leg_handles, _ = ax.get_legend_handles_labels()
    leg_labels = [el.name for el in config.data.models]
    leg = ax.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="center",
        title="",
        bbox_to_anchor=(-0.2, 1.1, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=-0.4,
        ncol=len(config.data.val_splits),
    )
    # Set X-label
    ax.set_xlabel("Inference time [s]")
    # X-lim
    ax.set_xlim(left=0.0, right=ax.get_xlim()[1])

    for ax in axes:

        # Change spines
        sns.despine(ax=ax, left=True, bottom=True)

        # Set Y-label
        ax.set_ylabel(None)

        # Y-tick labels
        ax.set_yticklabels(
            list([el.name for el in config.data.architectures]), fontsize="medium"
        )

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
    return fig


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Determine output dir
    if config.io.output_dir.upper() == "SLURM_TMPDIR":
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(to_absolute_path(config.io.output_dir))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    # Store args
    #     output_yml = output_dir / "phast_summary_plot.yml"
    #     with open(output_yml, "w") as f:
    #         yaml.dump(vars(config), f)
    # Read CSVs
    df_orig = pd.read_csv(to_absolute_path(config.io.input_csv_seeds), index_col=False)
    df_orig_times = pd.read_csv(
        to_absolute_path(config.io.input_csv_times), index_col=False
    )
    # Pre-process DataFrame
    df_orig = preprocess_df(df_orig, config.data)
    # Prepare data frames for plotting
    df_mae, df_time = make_plot_dataframes(df_orig, df_orig_times)
    # Plot
    fig = plot(df_orig, df_mae, df_time, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()
