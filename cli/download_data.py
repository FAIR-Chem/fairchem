import glob
import logging
import os
from typing import Dict, Optional

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS_s2ef: Dict[str, Dict[str, str]] = {
    "s2ef": {
        "200k": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
        "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
        "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
        "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz",
        "rattled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar",
        "md": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar",
    },
}

DOWNLOAD_LINKS_is2re: Dict[str, str] = {
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
}

S2EF_COUNTS = {
    "s2ef": {
        "200k": 200000,
        "2M": 2000000,
        "20M": 20000000,
        "all": 133934018,
        "val_id": 999866,
        "val_ood_ads": 999838,
        "val_ood_cat": 999809,
        "val_ood_both": 999944,
        "rattled": 16677031,
        "md": 38315405,
    },
}


def get_data(
    datadir: str,
    task: str,
    split: Optional[str],
    del_intmd_files: bool,
    num_workers: int = 1,
    # Only used in preprocessing
    get_edges: bool = False,
    ref_energy: bool = False,
    test_data: bool = False,
) -> None:
    os.makedirs(datadir, exist_ok=True)

    if task == "s2ef" and split is None:
        raise NotImplementedError("S2EF requires a split to be defined.")

    download_link: Optional[str] = None
    if task == "s2ef":
        assert (
            split is not None
        ), "Split must be defined for the s2ef dataset task"
        assert (
            split in DOWNLOAD_LINKS_s2ef[task]
        ), f'S2EF "{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS_s2ef["s2ef"].keys())}'
        download_link = DOWNLOAD_LINKS_s2ef[task][split]
    elif task == "is2re":
        download_link = DOWNLOAD_LINKS_is2re[task]
    else:
        raise Exception(f"Unrecognized task {task}")
    assert download_link is not None

    os.system(f"wget {download_link} -P {datadir}")
    filename = os.path.join(datadir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {datadir}")
    dirname = os.path.join(
        datadir,
        os.path.basename(filename).split(".")[0],
    )
    if task == "s2ef" and split != "test":
        assert (
            split is not None
        ), "Split must be defined for the s2ef dataset task"
        compressed_dir = os.path.join(dirname, os.path.basename(dirname))
        if split in ["200k", "2M", "20M", "all", "rattled", "md"]:
            output_path = os.path.join(datadir, task, split, "train")
        else:
            output_path = os.path.join(datadir, task, "all", split)
        uncompressed_dir = uncompress_data(compressed_dir, num_workers)
        preprocess_data(
            uncompressed_dir=uncompressed_dir,
            output_path=output_path,
            num_workers=num_workers,
            get_edges=get_edges,
            ref_energy=ref_energy,
            test_data=test_data,
        )

        verify_count(output_path, task, split)
    if task == "s2ef" and split == "test":
        os.system(f"mv {dirname}/test_data/s2ef/all/test_* {datadir}/s2ef/all")
    elif task == "is2re":
        os.system(f"mv {dirname}/data/is2re {datadir}")

    if del_intmd_files:
        cleanup(filename, dirname)


def uncompress_data(compressed_dir: str, num_workers: int) -> str:
    from cli import uncompress

    opdir = os.path.dirname(compressed_dir) + "_uncompressed"
    uncompress.main(
        ipdir=compressed_dir,
        opdir=opdir,
        num_workers=num_workers,
    )
    return opdir


def preprocess_data(
    uncompressed_dir: str,
    output_path: str,
    num_workers: int,
    get_edges: bool,
    ref_energy: bool,
    test_data: bool,
) -> None:
    from cli import preprocess_ef as preprocess

    preprocess.main(
        data_path=uncompressed_dir,
        out_path=output_path,
        num_workers=num_workers,
        get_edges=get_edges,
        ref_energy=ref_energy,
        test_data=test_data,
    )


def verify_count(output_path: str, task: str, split: str) -> None:
    paths = glob.glob(os.path.join(output_path, "*.txt"))
    count = 0
    for path in paths:
        lines = open(path, "r").read().splitlines()
        count += len(lines)
    assert (
        count == S2EF_COUNTS[task][split]
    ), f"S2EF {split} count incorrect, verify preprocessing has completed successfully."


def cleanup(filename: str, dirname: str) -> None:
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if os.path.exists(dirname + "_uncompressed"):
        shutil.rmtree(dirname + "_uncompressed")
