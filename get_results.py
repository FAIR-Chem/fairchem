import os
from pathlib import Path
from typing import List
import pandas as pd


# COLS = list("params forces_mae forces_cos energy_mae energy_force_within_threshold time".split())
COLS = list("params forces_mae forces_cos energy_mae energy_force_within_threshold time gpu_mem".split())


def main(fls: List[Path], feats: List[str]):
    results = []

    for fl in fls:
        text = fl.read_text().splitlines()
        # config = {
        #     feat: next(ln.strip().split()[-1] for ln in text if feat in ln.strip())
        #     for feat in feats
        # }
        config = {}
        for feat in feats:
            vals = [ln.strip().split()[-1] for ln in text if feat in ln.strip()]
            config[feat] = 1 if not vals else vals[0]

        param_line = [ln for ln in text if "parameters" in ln][0]
        params = param_line.strip().split()[-2]
        config["fl"] = fl
        config["params"] = params
        config["name"] = fl.parent.name.split("_")[0]

        memlines = [ln for ln in text if "[MemStats]" in ln]
        if len(memlines) > 0:
            memln = memlines[-1]
            mem = memln.split()[5].strip()
            mem = int(mem.strip(",").strip("G"))
            config["gpu_mem"] = mem

        try:
            metrics = text[-3].strip()
            # metrics = text[-3].strip().split(" ", 1)[1]
            # metrics = [ln for ln in text if "Validation:" in ln][-1].strip().split(" ", 1)[1]
            metrics = dict(x.split(":") for x in metrics.split(","))
            metrics = {k.strip(): v for k, v in metrics.items()}
            config.update(metrics)

            try:
                tm = float(text[-2].strip().split()[-1]) / 3600.
                config["time"] = tm
            except:
                config["time"] = None

            results.append(config)
        except:
            raise
    
    results = pd.DataFrame(results)
    results = results[feats + COLS]
    # print(results.to_string(index=None).replace("   ", "\t"))
    results.sort_values(by=feats)
    print(results.to_csv(index=False).replace(" ", "").replace(",", "\t"))
    
    # for fl in fls:
    #     text = fl.read_text().splitlines()
    #     feat_lines = [
    #         ";".join(ln.strip() for ln in text if feat in ln.strip())
    #         for feat in feats
    #     ]
    #     print(fl)
    #     print(" ".join(feat_lines))
    #     print(text[-3])
    #     print(text[-2])
    #     print()


if __name__ == "__main__":
    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp1/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["out_emb_channels", "hidden_channels"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp2/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["num_after_skip", "num_before_skip"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp3/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["int_emb_size"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp4/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["num_blocks"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp5/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["batch_size"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp6/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["num_spherical", "num_radial"]

    # fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp7/slurm/logs/").glob("dpp_*/*_0_log.out"))
    # feats = ["cutoff", "max_num_neighbors"]

    # fls = list(Path("exp/paralleldpp/pardpp.sweeps/sweep1/slurm/logs/").glob("*/*_0_log.out"))
    # feats = ["gpus_per_task", "num_blocks", "batch_size"]

    # fls = list(Path("exp/paralleldpp/pardpp.sweeps/sweep2/slurm/logs/").glob("*/*_0_log.out"))
    # feats = ["gpus_per_task", "hidden_channels", "out_emb_channels", "batch_size"]

    # fls = list(Path("exp/paralleldpp/pardpp.sweeps/sweep3/slurm/logs/").glob("*/*_0_log.out"))
    # feats = ["gpus_per_task", "int_emb_size", "batch_size"]

    # fls = list(Path("exp/paralleldpp/pardpp.sweeps/sweep4/slurm/logs/").glob("*/*_0_log.out"))
    # feats = ["gpus_per_task", "num_after_skip", "num_output_layers", "batch_size"]

    fls = list(Path("exp/paralleldpp/pardpp.sweeps/sweep5/slurm/logs/").glob("*/*_0_log.out"))
    feats = ["gpus_per_task", "cutoff", "num_neighbors", "batch_size"]

    main(fls, feats)

