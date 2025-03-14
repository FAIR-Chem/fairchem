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


def change_path_for_pypi(files_to_download: list[Path], par_dir: str) -> list[Path]:
    """
    Modify or exclude files from download if running in a PyPi-installed
    build.

    Installation of FAIR-Chem with PyPi does not include the entire
    directory structure of the fairchem repo. As such, files outside of
    `src` can't be downloaded and those in `src` should actually be in the
    `site-packages` directory. If a user wants these files, they must
    build from the git repo.

    :param files_to_download: List of files to be downloaded
    :param par_dir: the parent directory of the PyPi build,
                    probably "site-packages"
    :return: modified list of files to be downloaded
    """
    new_files = []
    for file in files_to_download:
        if (
            str(file.parents[len(file.parents) - 2]) != "src"
        ):  # no negative index in Python 3.9
            continue
        new_files.append(Path(str(file).replace("src", par_dir, 1)))
    return new_files


def download_file_group(file_group: str) -> None:
    """
    Download the given file group.

    :param file_group: Name of group of files to download
    """
    if file_group in FILE_GROUPS:
        files_to_download = FILE_GROUPS[file_group]
    elif file_group == "ALL":
        files_to_download = [item for group in FILE_GROUPS.values() for item in group]
    else:
        raise ValueError(
            f'Requested file group {file_group} not recognized. Please select one of {["ALL", *list(FILE_GROUPS)]}'
        )

    fc_root = fairchem_root()
    install_dir = fc_root.parents[1]
    fc_parent = str(fc_root.parent.name)
    if fc_parent != "src":
        files_to_download = change_path_for_pypi(files_to_download, fc_parent)

    missing_path = False
    for file in files_to_download:
        if not (install_dir / file.parent).exists():
            print(f"Cannot download {file}, the path does not exist.")
            missing_path = True
        elif not (install_dir / file).exists():
            print(f"Downloading {file}...")
            urlretrieve(S3_ROOT + file.name, install_dir / file)
        else:
            print(f"{file} already exists")

    if missing_path:
        print(
            "\n\nSome files could not be downloaded because their "
            "expected landing spot didn't exist. If you installed with "
            "PyPi perhaps additional packages like `fairchem-data-oc` "
            "are needed.\nIf all PyPi packages have been installed, you "
            "may need to install from the git repo to have the full "
            "fairchem contents."
        )


if __name__ == "__main__":
    args = parse_args()
    download_file_group(args.file_group)
