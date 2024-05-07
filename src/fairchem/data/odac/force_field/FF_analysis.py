import json

import numpy as np
import pylab as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error


# reading data ---------------------------------------------------------
def get_data(infile, limit=2):  # infile consists of lists of dictionaries
    with open(infile) as json_file:
        data_oms = json.load(json_file)

    int_DFT_CO2 = []
    int_DFT_H2O = []
    err_CO2 = []
    err_H2O = []
    raw_err_CO2 = []
    raw_err_H2O = []
    int_FF_CO2 = []
    int_FF_H2O = []
    for pt in data_oms:
        DFT = pt["y"]
        FF = pt["ff"]
        n = pt["name"]
        err = np.abs(DFT - FF)
        raw_err = FF - DFT
        if (
            np.abs(DFT) < limit and np.abs(FF) < limit
        ):  # only consider points in the specific range
            if "CO2" in n:
                int_DFT_CO2.append(DFT)
                err_CO2.append(err)
                raw_err_CO2.append(raw_err)
                int_FF_CO2.append(FF)
            if "H2O" in n:
                int_DFT_H2O.append(DFT)
                err_H2O.append(err)
                raw_err_H2O.append(raw_err)
                int_FF_H2O.append(FF)

    return (
        int_DFT_CO2,
        err_CO2,
        raw_err_CO2,
        int_FF_CO2,
        int_DFT_H2O,
        err_H2O,
        raw_err_H2O,
        int_FF_H2O,
    )


# util functions -------------------------------------------------------
def binned_average(DFT_ads, pred_err, bins):
    bin0 = -1000
    avgs = []
    for i, bin in enumerate(bins):
        if i == 0:
            left = bin0
        else:
            left = bins[i - 1]

        bin_errs = []
        for DFT, pred in zip(
            DFT_ads, pred_err
        ):  # this is a horribly inefficient way to do this...
            if DFT > left and DFT < bin:
                bin_errs.append(pred)
        if bin_errs:
            bin_avg = np.mean(bin_errs)
        else:
            bin_avg = 0
        avgs.append(bin_avg)
    return avgs


def bin_plot(
    ax, bins, heights, **kwargs
):  # stolen from https://stackoverflow.com/questions/36192074/manual-histogram-plot-in-python
    bins = list(bins)
    x1 = bins[:-1]
    x2 = bins[1:]

    w = np.array(x2) - np.array(x1)

    ax.bar(x1, heights, width=w, align="edge", edgecolor="black", **kwargs)


# plotting functions ---------------------------------------------------
def get_Fig4a(raw_error_CO2, raw_error_H2O, b=20, outfile="Fig5a.png"):
    # collect very low and high energies in one bin
    for i in range(len(raw_error_CO2)):
        if raw_error_CO2[i] < -1:
            raw_error_CO2[i] = -1.05
        if raw_error_CO2[i] > 1:
            raw_error_CO2[i] = 1.05

    for i in range(len(raw_error_H2O)):
        if raw_error_H2O[i] < -1:
            raw_error_H2O[i] = -1.05
        if raw_error_H2O[i] > 1:
            raw_error_H2O[i] = 1.05

    # plotting histogram
    plt.figure(figsize=(10, 7))
    plt.hist(
        raw_error_CO2,
        density=False,
        edgecolor="black",
        alpha=0.5,
        bins=b,
        color="crimson",
        label="CO$_{2}$",
    )
    plt.hist(
        raw_error_H2O,
        density=False,
        edgecolor="black",
        alpha=0.5,
        bins=b,
        color="dodgerblue",
        label="H$_{2}$O",
    )
    plt.xlabel("$E_{{int}}^{{FF}}$ – $E_{{int}}^{{DFT}}$ [eV]", fontsize=20)
    plt.ylabel("Number of configurations", fontsize=20)
    labels = [
        "<-1.00",
        "-0.75",
        "-0.50",
        "-0.25",
        "0.00",
        "0.25",
        "0.50",
        "0.75",
        ">1.00",
    ]
    label_pos = np.linspace(-1 + 1 / b, 1 - 1 / b, len(labels))
    plt.xticks(label_pos, labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(("CO$_{2}$", "H$_{2}$O"), fontsize=16)
    plt.savefig(outfile)


def get_Fig4b(int_DFT_CO2, err_CO2, int_DFT_H2O, err_H2O, outfile="Fig5b.png"):
    E_min = -2
    E_max = 2
    bins = np.linspace(E_min, E_max, 20)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax2 = ax.twinx()

    avgs_CO2 = binned_average(int_DFT_CO2, err_CO2, bins)
    avgs_H2O = binned_average(int_DFT_H2O, err_H2O, bins)

    bin_plot(ax, bins, avgs_CO2[1:], color="crimson", alpha=0.5)
    bin_plot(ax, bins, avgs_H2O[1:], color="dodgerblue", alpha=0.5)

    density = gaussian_kde(int_DFT_CO2)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    x = np.linspace(E_min, E_max, 100)
    ax2.plot(x, density(x), color="crimson")

    density = gaussian_kde(int_DFT_H2O)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    x = np.linspace(E_min, E_max, 100)
    ax2.plot(x, density(x), color="dodgerblue")

    ax.set_xlabel("DFT interaction energy [eV]", fontsize=20)
    ax.set_ylabel("Average error within bin [eV]", fontsize=20)
    ax.tick_params(labelsize=16)
    ax2.set_ylabel("Density of points", fontsize=20)
    ax2.tick_params(labelsize=20)
    ax.legend(("CO$_{2}$", "H$_{2}$O"), fontsize=16, loc="upper left")

    fig.savefig(outfile)


def get_Fig4c(DFT_CO2, err_CO2, outfile="Fig5c.png"):
    xy = np.vstack([DFT_CO2, err_CO2])
    z = gaussian_kde(xy, bw_method=1)(xy)

    DFT_CO2 = np.array(DFT_CO2)
    err_CO2 = np.array(err_CO2)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = DFT_CO2[idx], err_CO2[idx], z[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [(1, 0.65, 0.65), (0.5, 0, 0)]  # Light red to dark red
    custom_cmap = LinearSegmentedColormap.from_list("custom_reds", colors, N=256)
    scatter = ax.scatter(
        x_sorted, y_sorted, c=z_sorted, cmap=custom_cmap, s=25, alpha=0.5
    )

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax)
    cbar.ax.set_ylabel("Point density", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax.margins(y=0)
    custom_y_ticks = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    custom_y_labels = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
    ax.set_yticks(custom_y_ticks, custom_y_labels)
    custom_x_ticks = [-2, -1, 0, 1, 2]
    custom_x_labels = ["-2", "-1", "0", "1", "2"]
    ax.set_xticks(custom_x_ticks, custom_x_labels)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("$E_{{int}}^{{DFT}}$ [eV]", fontsize=20)
    ax.set_ylabel("|$E_{{int}}^{{FF}}$ – $E_{{int}}^{{DFT}}$| [eV]", fontsize=20)
    plt.savefig(outfile)


def get_Fig4d(DFT_H2O, err_H2O, outfile="Fig5d.png"):
    xy = np.vstack([DFT_H2O, err_H2O])
    z = gaussian_kde(xy, bw_method=0.4)(xy)

    DFT_H2O = np.array(DFT_H2O)
    err_H2O = np.array(err_H2O)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = DFT_H2O[idx], err_H2O[idx], z[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [(0.65, 0.65, 1), (0, 0, 0.5)]  # Light blue to dark blue
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)
    scatter = ax.scatter(
        x_sorted, y_sorted, c=z_sorted, cmap=custom_cmap, s=25, alpha=0.5
    )

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax)
    cbar.ax.set_ylabel("Point density", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax.margins(y=0)
    custom_y_ticks = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    custom_y_labels = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
    ax.set_yticks(custom_y_ticks, custom_y_labels)
    custom_x_ticks = [-2, -1, 0, 1, 2]
    custom_x_labels = ["-2", "-1", "0", "1", "2"]
    ax.set_xticks(custom_x_ticks, custom_x_labels)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("$E_{{int}}^{{DFT}}$ [eV]", fontsize=20)
    ax.set_ylabel("|$E_{{int}}^{{FF}}$ – $E_{{int}}^{{DFT}}$| [eV]", fontsize=20)
    plt.savefig(outfile)


# error calculations ---------------------------------------------------
def phys_err(DFT, FF):
    # physisorption error
    phys_FF = []
    phys_DFT = []
    phys_FF_lst = []
    for i in range(len(DFT)):
        if DFT[i] <= 0 and DFT[i] >= -0.5:
            phys_DFT.append(DFT[i])
            phys_FF.append(FF[i])
            if np.abs(FF[i]) > 1:
                phys_FF_lst.append(FF[i])
    return mean_absolute_error(phys_DFT, phys_FF)


def chem_err(DFT, FF):
    # chemisorption error
    DAC_DFT = []
    DAC_FF = []
    for i in range(len(DFT)):
        if DFT[i] <= -0.5 and DFT[i] >= -2:
            DAC_DFT.append(DFT[i])
            DAC_FF.append(FF[i])
    return mean_absolute_error(DAC_DFT, DAC_FF)


# main -----------------------------------------------------------------
if __name__ == "__main__":
    infile = "/storage/home/hcoda1/8/lbrabson3/p-amedford6-0/s2ef/final/data_w_oms.json"
    DFT_CO2, err_CO2, rerr_CO2, FF_CO2, DFT_H2O, err_H2O, rerr_H2O, FF_H2O = get_data(
        infile
    )
    get_Fig4a(rerr_CO2, rerr_H2O)
    get_Fig4b(DFT_CO2, err_CO2, DFT_H2O, err_H2O)
    get_Fig4c(DFT_CO2, err_CO2)
    get_Fig4d(DFT_H2O, err_H2O)

    print("Overall MAE: {} eV".format(np.mean(err_CO2 + err_H2O)))
    print("CO2 error: {} eV".format(np.mean(err_CO2)))
    print("H2O error: {} eV".format(np.mean(err_H2O)))
    print(
        "Overall physisorption error: {} eV".format(
            phys_err(DFT_CO2 + DFT_H2O, FF_CO2 + FF_H2O)
        )
    )
    print("CO2 physisorption error: {} eV".format(phys_err(DFT_CO2, FF_CO2)))
    print("H2O physisorption error: {} eV".format(phys_err(DFT_H2O, FF_H2O)))
    print(
        "Overall chemisorption error: {} eV".format(
            chem_err(DFT_CO2 + DFT_H2O, FF_CO2 + FF_H2O)
        )
    )
    print("CO2 chemisorption error: {} eV".format(chem_err(DFT_CO2, FF_CO2)))
    print("H2O chemisorption error: {} eV".format(chem_err(DFT_H2O, FF_H2O)))
