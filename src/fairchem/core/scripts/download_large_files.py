from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from fairchem.core.common.tutorial_utils import fairchem_root

S3_ROOT = "https://dl.fbaipublicfiles.com/opencatalystproject/data/large_files/"

FILE_GROUPS = {
    "odac": [
        Path("configs/odac/s2ef/scaling_factors/painn.pt"),
        Path("src/fairchem/data/odac/force_field/data_w_oms.json"),
        Path(
            "src/fairchem/data/odac/promising_mof/promising_mof_features/JmolData.jar"
        ),
        Path(
            "src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt"
        ),
        Path("src/fairchem/data/odac/supercell_info.csv"),
    ],
    "oc": [Path("src/fairchem/data/oc/databases/pkls/bulks.pkl")],
    "adsorbml": [
        Path(
            "src/fairchem/applications/AdsorbML/adsorbml/2023_neurips_challenge/oc20dense_mapping.pkl"
        ),
        Path(
            "src/fairchem/applications/AdsorbML/adsorbml/2023_neurips_challenge/ml_relaxed_dft_targets.pkl"
        ),
    ],
    "cattsunami": [
        Path("tests/applications/cattsunami/tests/autoframe_inputs_dissociation.pkl"),
        Path("tests/applications/cattsunami/tests/autoframe_inputs_transfer.pkl"),
    ],
    "docs": [
        Path("docs/tutorials/NRR/NRR_example_bulks.pkl"),
        Path("docs/core/fine-tuning/supporting-information.json"),
        Path("docs/core/data.db"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_group",
        type=str,
        help="Group of files to download",
        default="ALL",
        choices=["ALL", *list(FILE_GROUPS)],
    )
    return parser.parse_args()


def download_file_group(file_group):
    if file_group in FILE_GROUPS:
        files_to_download = FILE_GROUPS[file_group]
    elif file_group == "ALL":
        files_to_download = [item for group in FILE_GROUPS.values() for item in group]
    else:
        raise ValueError(
            f'Requested file group {file_group} not recognized. Please select one of {["ALL", *list(FILE_GROUPS)]}'
        )

    fc_root = fairchem_root().parents[1]
    for file in files_to_download:
        if not (fc_root / file).exists():
            print(f"Downloading {file}...")
            urlretrieve(S3_ROOT + file.name, fc_root / file)
        else:
            print(f"{file} already exists")


if __name__ == "__main__":
    args = parse_args()
    download_file_group(args.file_group)
