import re
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from minydra import resolved_args
from tqdm import tqdm

if __name__ == "__main__":

    args = resolved_args()

    assert args.file is not None
    assert Path(args.file).exists()
    assert Path(args.file).is_file()

    with open(args.file, "r") as f:
        lines = f.read()

    samples = [
        s
        for s in lines.split(
            "------------------------------\n------------------------------"
        )
        if "Actions to Data" in s and "ABORTING" not in s
    ]

    times = defaultdict(list)
    metadatas = []
    time_regex = re.compile(r"(.*) \| Done! \((.*)s\)")
    total_adsorbed_regex = re.compile(r"Total adsorbed_surfaces: (\d+)")
    non_reasonable_regex = re.compile(r"Non reasonable configs: (\d+)/(\d+)")

    metadata_regexs = {
        "adsorbate_id": re.compile(
            r"args(?:\.actions|)\.adsorbate_id is None, choosing (\d+)"
        ),
        "adsorbate_desc": re.compile(r"# Selected adsorbate: (.+)"),
        "bulk_id": re.compile(r"args\.actions\.bulk_id is None, choosing (\d+)"),
        "bulk_desc": re.compile(r"# Selected bulk: (.+)"),
        "surface_id": re.compile(r"args\.actions\.surface_id is None, choosing (\d+)"),
        "surface_desc": re.compile(r"# Selected surface: (.+)"),
        "bond_indices": re.compile(r"bond_indices: (.+)"),
    }

    keys = []
    time_keys = []
    for s, sample in tqdm(enumerate(samples), total=len(samples)):
        metadatas.append({})
        matches = time_regex.findall(sample)
        if not time_keys:
            time_keys = set([k.strip() for k, _ in matches] + ["Actions to Data"])
        matches += [
            (
                "Total adsorbed_surfaces",
                int(total_adsorbed_regex.findall(sample)[0]),
            ),
            (
                "Proportion of non reasonable adsorbed_surfaces",
                float(non_reasonable_regex.findall(sample)[0][0])
                / float(non_reasonable_regex.findall(sample)[0][1]),
            ),
        ]

        for k, v in matches:
            k = k.strip()
            if "Actions to Data" in k:
                k = "Actions to Data"
            times[k].append(float(v))
            metadatas[-1][k] = float(v)
            if s == 0:
                keys.append(k)
        for name, reg in metadata_regexs.items():
            meta = reg.findall(sample)[0]
            if "id" in name:
                meta = int(meta)
            metadatas[-1][name] = meta

    means = {k: m for k, v in times.items() if ((m := np.mean(v)) > 0.1)}
    stds = {k: np.std(v) for k, v in times.items() if k in means}
    keys = [k for k in keys if k in means]

    cmap = plt.get_cmap("viridis")
    cnorm = colors.Normalize(vmin=0, vmax=len(samples))
    scalar_map = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    n_plots = len(means.keys())

    ncols = args.plot_ncols or 3
    nrows = n_plots // ncols
    if n_plots % ncols != 0:
        nrows += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i, k in tqdm(enumerate(keys), total=len(keys)):
        ax = axs.flat[i]
        bars = ax.bar(range(len(times[k])), times[k])
        for b, bar in enumerate(bars):
            bar.set_color(scalar_map.to_rgba(b))

        title = k
        if k in time_keys:
            title += f" ({means[k]:.2f}s +/- {stds[k]:.2f}s)"
        else:
            title += " (count)"

        ax.set_title(title, fontsize=8)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)

    plt.suptitle(f"Time (s) for operations or printed counts ({len(samples)} samples)")

    plt.savefig(args.out_png or f"{Path(args.file).stem}.png", dpi=150)
    with open(args.out_json or f"{Path(args.file).stem}.json", "w") as f:
        json.dump(metadatas, f)
