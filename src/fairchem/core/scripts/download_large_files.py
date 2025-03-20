from __future__ import annotations

import argparse
import logging
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


def change_path_for_pypi(
    files_to_download: list[Path],
    par_dir: str,
    install_dir: Path,
    test_par_dir: Path | None,
) -> list[Path]:
    """
    Modify or exclude files from download if running in a PyPi-installed
    build.

    Installation of FAIR-Chem with PyPi does not include the entire
    directory structure of the fairchem repo. As such, files outside of
    `src` can't be downloaded and those in `src` should actually be in the
    `site-packages` directory. If a user wants these files, they must
    build from the git repo.

    If the tests have been separately downloaded (e.g. from the git repo),
    then we can download if we've been told where those tests have been
    downloaded to. Note that we can't divine that location from anything
    in fairchem.core because they would have to be somewhere "unexpected"
    since we've built with PyPi which shouldn't have tests at all.

    :param files_to_download: List of files to be downloaded
    :param par_dir: the parent directory of the PyPi build,
                    probably "site-packages"
    :param install_dir: path to where fairchem.core was installed
    :param test_par_dir: path to where tests have been downloaded
                         (not necessarily the same as install_dir)
    :return: modified list of files to be downloaded
    """
    new_files = []
    for file in files_to_download:
        # We check the top-level name of the file so we know if it has a
        # home in PyPi builds
        top_level_name = file.parts[0]
        if top_level_name == "tests" and test_par_dir is not None:
            new_files.append(test_par_dir / file)
        elif top_level_name == "src":
            # turn `src` into `site-packages` or whatever the correct name is
            new_files.append(install_dir / Path(str(file).replace("src", par_dir, 1)))
    return new_files


def download_file_group(file_group: str, test_par_dir: Path | None = None) -> None:
    """
    Download the given file group.

    :param file_group: Name of group of files to download
    :param test_par_dir: Parent directory where fairchem tests have been
                         downloaded
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
    # The grandparent of fairchem_root is the install location,
    # this is the git repo top-level directory for dev builds, i.e. `fairchem`
    install_dir = fc_root.parent.parent
    # The parent of fairchem_root is `src` on git repo/dev builds and
    # usually `site-packages` for PyPi builds (to be safe, we don't assume
    # that name)
    fc_parent = str(fc_root.parent.name)
    if fc_parent != "src":
        files_to_download = change_path_for_pypi(
            files_to_download, fc_parent, install_dir, test_par_dir
        )
    else:
        files_to_download = [install_dir / file for file in files_to_download]

    missing_path = False
    for file in files_to_download:
        if not file.parent.exists():
            print(f"Cannot download {file}, the path does not exist.")
            missing_path = True
        elif not file.exists():
            print(f"Downloading {file}...")
            urlretrieve(S3_ROOT + file.name, file)
        else:
            print(f"{file} already exists")

    if missing_path:
        logging.error(
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
