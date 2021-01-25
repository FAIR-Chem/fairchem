import os
from pathlib import Path
from typing import List
import pandas as pd


COLS = list("params forces_mae forces_cos energy_mae energy_force_within_threshold time".split())


def main(fls: List[Path], feats: List[str]):
    results = []

    for fl in fls:
        text = fl.read_text().splitlines()
        config = {
            feat: next(ln.strip().split()[-1] for ln in text if feat in ln.strip())
            for feat in feats
        }
        param_line = [ln for ln in text if "parameters" in ln][0]
        params = param_line.strip().split()[-2]
        config["fl"] = fl
        config["params"] = params

        try:
            # metrics = text[-3].strip().split(" ", 1)[1]
            metrics = [ln for ln in text if "Validation:" in ln][-1].strip().split(" ", 1)[1]
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
            pass
    
    results = pd.DataFrame(results)
    results = results[feats + COLS]
    # print(results.to_string(index=None).replace("   ", "\t"))
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

    fls = list(Path("exp/dpp_sweep/dpp_2M/sweep_dpp7/slurm/logs/").glob("dpp_*/*_0_log.out"))
    feats = ["cutoff", "max_num_neighbors"]


    main(fls, feats)

