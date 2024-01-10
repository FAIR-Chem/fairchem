import sys
from pathlib import Path
from minydra import resolved_args
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.datasets.lmdb_dataset import DeupDataset
from ocpmodels.common.utils import set_cpus_to_workers
from ocpmodels.common.data_parallel import ParallelCollater


def clipped_geomspace(min_val, max_val, min_log, n_lin, n_log):
    lin = np.linspace(min_val, min_log, n_lin)
    log = np.geomspace(min_log, max_val, n_log)
    return np.concatenate((lin, log))


def symmetric_clipped_geomspace(x, min_log, n_lin, n_log):
    x = np.array(x)
    n_lin = int(n_lin)
    n_log = int(n_log)
    pos = x[x > 0]
    neg = x[x < 0]
    pos = clipped_geomspace(0, pos.max() * 1.1, min_log, n_lin, n_log)
    neg = -clipped_geomspace(0, -neg.min() * 1.1, min_log, n_lin, n_log)[::-1]
    return sorted(set(np.concatenate((neg, pos)).tolist()))


def scatter_hist(x, y, z, ax, ax_histx, ax_histy, logx, logy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # log scale
    ax_histx.set_yscale("log")
    ax_histy.set_xscale("log")

    # the scatter plot:
    ax.scatter(x, y, s=1, alpha=0.2, c=z, cmap="viridis")

    n_bins = 250
    if logx == "symlog":
        x_bins = symmetric_clipped_geomspace(x, 1e-1, 25, 150)
    elif logx == "log":
        x_bins = np.logspace(np.log10(min(x)), np.log10(max(x)), n_bins)
    else:
        x_bins = np.linspace(min(x), max(x), n_bins)

    if logy == "symlog":
        y_bins = symmetric_clipped_geomspace(y, 1e-1, 25, 150)
    elif logy == "log":
        y_bins = np.logspace(np.log10(min(y)), np.log10(max(y)), n_bins)
    else:
        y_bins = np.linspace(min(y), max(y), n_bins)

    ax_histx.hist(
        x,
        bins=x_bins,
        alpha=0.5,
    )
    ax_histy.hist(
        y,
        bins=y_bins,
        orientation="horizontal",
        alpha=0.5,
    )
    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())


if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "ds_path": "/home/mila/s/schmidtv/scratch/ocp/runs/3264530/deup_dataset",
            "deup_str": "deup-train-val_id",
            "batch_size": 512,
            "max_batches": -1,
        }
    )

    base_ds_config = {
        "train": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/train/"
        },
        "val_id": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_id/"
        },
        "val_ood_cat": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_cat/"
        },
        "val_ood_ads": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_ads/"
        },
        "val_ood_both": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_both/"
        },
    }

    ds_config = {
        args.deup_str: {
            "src": args.ds_path,
        },
        **base_ds_config,
    }

    ds = DeupDataset(ds_config, args.deup_str)
    cpus = set_cpus_to_workers({"optim": {}, "silent": True})["optim"]["num_workers"]
    parallel_collater = ParallelCollater(0, True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=cpus,
        collate_fn=parallel_collater,
    )

    data = {}
    keys = ["deup_energy_pred_mean", "deup_energy_pred_std", "deup_loss"]
    n_samples = len(ds)
    if args.max_batches > 0:
        n_samples = args.batch_size * args.max_batches

    for i, batch in enumerate(
        tqdm(loader, total=len(loader) if args.max_batches <= 0 else args.max_batches)
    ):
        if i == args.max_batches:
            break
        batch = batch[0]
        for key in keys:
            if key not in data:
                data[key] = []
            data[key].extend(batch[key].tolist())

    plot_data = [
        {
            "x": data["deup_energy_pred_mean"],
            "y": data["deup_energy_pred_std"],
            "logx": "symlog",
            "logy": "log",
            "xlabel": "Energy prediction mean",
            "ylabel": "Energy prediction std",
        },
        {
            "x": data["deup_loss"],
            "y": data["deup_energy_pred_std"],
            "logx": "log",
            "logy": "log",
            "xlabel": "Loss",
            "ylabel": "Energy prediction std",
        },
    ]

    for pd in plot_data:
        x = pd["x"]
        y = pd["y"]
        logx = pd["logx"]
        logy = pd["logy"]
        xlabel = pd["xlabel"]
        ylabel = pd["ylabel"]

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax_histx = fig.add_subplot(
            gs[0, 0],
            sharex=ax,
        )
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        scatter_hist(x, y, None, ax, ax_histx, ax_histy, logx, logy)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(logx)
        ax.set_yscale(logy)
        # add grey grid on main plot:
        ax.grid(color="grey", linestyle="-", linewidth=0.30, alpha=0.5, which="both")
        # ax.tricontourf(x, y, z, levels=5, cmap="RdBu_r", alpha=0.5)
        fig.suptitle(f"n_samples: {n_samples} | {args.deup_str}")
        plt.savefig(
            "scatter_hist-"
            + "_".join([k.lower() for k in xlabel.split()])
            + "_".join([k.lower() for k in ylabel.split()])
            + ".png"
        )
    plt.close("all")
