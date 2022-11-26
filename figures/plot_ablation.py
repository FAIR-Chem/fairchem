"""
This script plots the results of the ablation study.
"""
import sys
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
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
from utils import get_palettes_methods_family, get_palette_val, get_palette_methods
from utils import plot_setup


def min_max_errorbar(a):
    return (np.min(a), np.max(a))


def make_plot_dataframes(df_orig):
    # Build data set
    df_mae = pd.DataFrame(
        columns=["architecture", "method-family", "method", "val", "mae_phast_impr"]
    )
    df_time = pd.DataFrame(
        columns=["architecture", "method-family", "method", "time", "phast_fraction"]
    )
    for method in df_orig.method.unique():
        if method == "graclus":
            continue
        for val in df_orig.val.unique():
            df_method_val = df_orig.loc[
                (df_orig.method == method) & (df_orig.val == val)
            ].sort_values("architecture")
            df_baseline_val = df_orig.loc[
                (df_orig.method == "baseline") & (df_orig.val == val)
            ].sort_values("architecture")
            # MAE
            mae_phast_impr = (
                -100
                * (df_method_val["mae"].values - df_baseline_val["mae"].values)
                / df_baseline_val["mae"].values
            )
            # Update df
            for arch, mae in zip(sorted(df_orig.architecture.unique()), mae_phast_impr):
                dfaux = pd.DataFrame.from_dict(
                    [
                        {
                            "architecture": arch,
                            "method": method,
                            "method-family": df_method_val["method-family"].values[0],
                            "val": val,
                            "mae_phast_impr": mae,
                        }
                    ]
                )
                df_mae = pd.concat([df_mae, dfaux], axis=0, ignore_index=True)
            # Time
            time_baseline_arr = df_baseline_val.sort_values("architecture")[
                "inf. time"
            ].values
            time_phast_arr = df_method_val.sort_values("architecture")[
                "inf. time"
            ].values
            time_phast_fraction_arr = 100 * (time_phast_arr / time_baseline_arr)
            # Update df
            for arch, time_baseline, time_phast, time_phast_fraction in zip(
                sorted(df_orig.architecture.unique()),
                time_baseline_arr,
                time_phast_arr,
                time_phast_fraction_arr,
            ):
                dfaux = pd.DataFrame.from_dict(
                    [
                        {
                            "architecture": arch,
                            "method": method,
                            "method-family": df_method_val["method-family"].values[0],
                            "model": "baseline",
                            "time": time_baseline,
                            "phast_fraction": 0.0,
                        }
                    ]
                )
                df_time = pd.concat([df_time, dfaux], axis=0, ignore_index=True)
                dfaux = pd.DataFrame.from_dict(
                    [
                        {
                            "architecture": arch,
                            "method": method,
                            "method-family": df_method_val["method-family"].values[0],
                            "model": "phast",
                            "time": time_phast,
                            "phast_fraction": time_phast_fraction,
                        }
                    ]
                )
                df_time = pd.concat([df_time, dfaux], axis=0, ignore_index=True)
    return df_mae, df_time


def plot(df_orig, df_mae, df_time, config):
    plot_setup()

    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=True, dpi=config.plot.dpi, figsize=(8, 8)
    )

    # MAE PhAST improvement
    ax = sns.pointplot(
        ax=axes[0],
        data=df_mae,
        estimator=np.mean,
        errorbar=min_max_errorbar,
        order=[el.key for el in config.data.methods],
        x="mae_phast_impr",
        y="method",
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
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    for arch in df_orig.architecture.unique():
        ax = sns.pointplot(
            ax=axes[0],
            data=df_mae.loc[df_mae["architecture"] == arch],
            estimator=np.mean,
            errorbar=min_max_errorbar,
            order=[el.key for el in config.data.methods],
            x="mae_phast_impr",
            y="method",
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
    # Legend
    leg_labels = [el.name for el in config.data.architectures]
    leg = ax.legend(
        handles=leg_handles[:3],
        labels=leg_labels,
        loc="center",
        title="",
        bbox_to_anchor=(-0.2, 1.05, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=-0.4,
        ncol=len(config.data.val_splits),
    )
    if config.plot.plot_baseline:
        # Plot Baseline
        df_mae_baseline = pd.DataFrame(
            {
                "method": [el.key for el in config.data.methods],
                "mae_phast_impr": [0.0 for _ in [el.key for el in config.data.methods]],
            }
        )
        ax = sns.pointplot(
            ax=axes[0],
            data=df_mae_baseline,
            order=[el.key for el in config.data.methods],
            x="mae_phast_impr",
            y="method",
            markers=config.plot.markers.models["baseline"],
            color=get_palette_methods(config.plot.colors.methods.palette)[0],
            errwidth=1.5,
            scale=0.6,
            join=False,
        )
    # Set X-label
    ax.set_xlabel("MAE improvement [%] per method")

    # Time PhAST improvement
    ax = sns.pointplot(
        ax=axes[1],
        data=df_time,
        estimator=np.mean,
        errorbar=min_max_errorbar,
        order=[el.key for el in config.data.methods],
        x="time",
        y="method",
        hue="model",
        hue_order=["baseline", "phast"],
        markers=[config.plot.markers.models[el.key] for el in config.data.models],
        palette=get_palette_methods(config.plot.colors.methods.palette),
        dodge=config.plot.dodge * 0.5,
        errwidth=config.plot.errwidth,
        scale=config.plot.scale,
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
        bbox_to_anchor=(-0.2, 1.05, 1.0, 0.0),
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
        sns.despine(left=True, bottom=True)

        # Set Y-label
        ax.set_ylabel(None)

        # Y-tick labels
        ax.set_yticklabels(
            list([el.name for el in config.data.methods]), fontsize="medium"
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
        _, y_bottom = display2data.transform(ax.transAxes.transform((0.0, 0.0)))
        _, y_top = display2data.transform(ax.transAxes.transform((0.0, 1.0)))
        ax.set_ylim(bottom=y_bottom, top=y_top)

        # Draw line at H0
        if ax == axes[0]:
            y = np.arange(ax.get_ylim()[1], ax.get_ylim()[0], 0.1)
            x = 0.0 * np.ones(y.shape[0])
            ax.plot(x, y, linestyle=":", linewidth=2.0, color="black", zorder=1)

        # Draw shaded areas
        shade_palette = get_palettes_methods_family(
            df_orig,
            [el.key for el in config.data.methods],
            config.plot.colors.methods_family.palette,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        width = np.abs(xlim[1] - xlim[0])
        offset = 0.05
        x0 = xlim[0] - offset * width
        width = width + 2 * offset * width
        ax.set_xlim(left=x0, right=x0 + width)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        height_alpha = 0.9
        height = height_alpha * 1.0 / len(config.data.methods)
        margin = (1.0 - height_alpha) / (len(config.data.methods) - 1)
        for idx in range(len(config.data.methods)):
            rect = mpatches.Rectangle(
                xy=(x0, height * idx + margin * idx),
                width=width,
                height=height,
                transform=trans,
                linewidth=0.0,
                edgecolor="none",
                facecolor=shade_palette[idx],
                alpha=0.15,
                zorder=0,
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
    df_orig = pd.read_csv(to_absolute_path(config.io.input_csv), index_col=False)
    # Prepare data frames for plotting
    df_mae, df_time = make_plot_dataframes(df_orig)
    # Plot
    fig = plot(df_orig, df_mae, df_time, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()
