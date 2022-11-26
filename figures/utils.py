import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import hsv_to_rgb

family_hue_start = {"atom-embeddings": 0.7, "energy-head": 0.3, "graph-creation": 0.0}
hue_dist = 0.05
sat = 0.3
val = 0.8

family_palette = {
    "atom-embeddings": "flare",
    "energy-head": "crest",
    "graph-creation": "magma",
}


def plot_setup():
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


def get_palette_val(palette_name, n_colors):
    return sns.color_palette(palette_name, as_cmap=False, n_colors=n_colors)


def get_palette_models(palette_name):
    palette = sns.color_palette(palette_name, as_cmap=False, n_colors=3)
    return [palette[2], palette[0]]


def get_palettes_methods_family_hue(df_all, dict_methods):
    df = df_all[["method", "method-family"]].value_counts().reset_index(name="count")
    n_per_fam = df["method-family"].value_counts().drop("baseline")
    palette = []
    idx = 0
    for method in dict_methods.keys():
        fam = df.loc[df["method"] == method]["method-family"].values[0]
        print(method, fam)
        hue = family_hue_start[fam] + idx * hue_dist
        color = tuple(hsv_to_rgb((hue, sat, val)))
        palette.append(color)
        idx += 1
        if idx == n_per_fam[fam]:
            idx = 0
    return palette


def get_palettes_methods_family_prev(df_all, methods):
    df = df_all[["method", "method-family"]].value_counts().reset_index(name="count")
    n_per_fam = df["method-family"].value_counts().drop("baseline")
    palette = []
    idx = 0
    for method in methods:
        fam = df.loc[df["method"] == method]["method-family"].values[0]
        fam_palette = sns.color_palette(family_palette[fam], as_cmap=False, n_colors=9)
        palette.append(fam_palette[idx])
        idx += 1
        if idx == n_per_fam[fam]:
            idx = 0
    return palette[::-1]


def get_palette_methods_family(df_all, methods, palette_name):
    df = df_all[["method", "method-family"]].value_counts().reset_index(name="count")
    n_per_fam = df["method-family"].value_counts().drop("baseline")
    palette_base = sns.color_palette(
        palette_name, as_cmap=False, n_colors=len(df["method-family"].unique())
    )
    palette_base = {
        fam: color for fam, color in zip(df["method-family"].unique(), palette_base)
    }
    palette = []
    idx = 0
    for method in methods:
        fam = df.loc[df["method"] == method]["method-family"].values[0]
        fam_palette = sns.color_palette(family_palette[fam], as_cmap=False, n_colors=9)
        palette.append(palette_base[fam])
        idx += 1
        if idx == n_per_fam[fam]:
            idx = 0
    return palette[::-1]
