"""
export base='/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/train'
python make_ads_metadata.py --lmdb_dir_path=$base/train --force
python make_ads_metadata.py --lmdb_dir_path=$base/val_id --force
python make_ads_metadata.py --lmdb_dir_path=$base/val_ood_cat --force
python make_ads_metadata.py --lmdb_dir_path=$base/val_ood_ads --force
python make_ads_metadata.py --lmdb_dir_path=$base/val_ood_both --force
"""
import bisect
import pickle
import tarfile
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import numpy as np
import lmdb
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm
import json


def download_tar(target_path, ads):
    if not target_path.exists():
        print("Downloading...", flush=True, end="")
        response = requests.get(ads["Downloadable path"], stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())
        print("Done!")
    else:
        print("Using existing tar file.")


def download_pkl(target_path, url):
    if not target_path.exists():
        # create the relevant folder if it does not exist
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True)
        print("Downloading...", flush=True, end="")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())
        print("Done!")
    else:
        print("Using existing pkl file.")


def untar(fp):
    parent = Path(fp).parent
    print("Untarring...", end="", flush=True)
    tar = tarfile.open(str(fp))
    parent.mkdir(parents=True)
    tar.extractall(path=str(parent))
    tar.close()
    print("Done!")


def pyg2_data_transform(data: Data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(**{k: v for k, v in data.__dict__.items() if v is not None})

    return data


def parse_ads_table(md):
    adsorbates = {}
    cols = []
    for l in md.splitlines():
        if l.startswith("|Adsorbate"):
            cols = [c.strip() for c in l.split("|") if c]
            adsorbates = {c: [] for c in cols}
        if l.startswith("|") and "https" in l:
            vals = [v.strip() for v in l.split("|") if v]
            assert len(vals) == len(cols)
            for v, c in zip(vals, cols):
                adsorbates[c].append(v)
    ads = {}
    for i in range(len(list(adsorbates.values())[0])):
        a = {col: adsorbates[col][i] for col in cols}
        ads[int(Path(a["Downloadable path"]).stem)] = a
    adsorbates = ads
    return adsorbates


def get_pickled_from_db(idx, keys, envs, keylen_cumulative):
    # Figure out which db this should be indexed from.
    db_idx = bisect.bisect(keylen_cumulative, idx)
    # Extract index of element within that db.
    el_idx = idx
    if db_idx != 0:
        el_idx = idx - keylen_cumulative[db_idx - 1]
    assert el_idx >= 0

    # Return features.
    return (
        db_idx,
        el_idx,
        pyg2_data_transform(
            pickle.loads(
                envs[db_idx].begin().get(f"{keys[db_idx][el_idx]}".encode("ascii"))
            )
        ),
    )


def connect_db(lmdb_path):
    # https://lmdb.readthedocs.io/en/release/#environment-class
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    return env


def indices_for_ads(metadata, ads):
    if isinstance(metadata, str):
        print("Loading metadata from file", str(metadata))
        metadata = json.loads(metadata)
    print("Finding indices for ads:", ads, "in dataset", metadata["source"][0])
    if isinstance(ads, str):
        ads = [ads]
    ads = set(ads)
    return [i for a, i in zip(metadata["ads_symbols"], metadata["ds_idx"]) if a in ads]


if __name__ == "__main__":
    data_mapping_url = (
        "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="/network/scratch/s/schmidtv/ocp/datasets/ocp/per_ads",
        help="Path to the directory where the metadata will be stored.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset. If not provided, will be asked for."
        + " If the dataset already exists, will ask for overwrite permission.",
    )
    parser.add_argument(
        "--lmdb_dir_path",
        type=str,
        default="/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/10k/train",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, will not ask for overwrite permission "
        + "if the dataset already exists.",
    )
    args = parser.parse_args()

    print("\n".join([f"• {k:20}: {v}" for k, v in vars(args).items()]))

    base_path = Path(args.base_path)
    lmdb_dir_path = Path(args.lmdb_dir_path)
    dataset_name = args.dataset_name
    force = args.force

    data_mapping_path = base_path / "oc20_data_mapping.pkl"
    output_file = base_path / f"{dataset_name}.json"
    db_paths = sorted(lmdb_dir_path.glob("*.lmdb"))

    assert len(db_paths) > 0, f"No LMDBs found in '{lmdb_dir_path}'"

    if not dataset_name:
        print("Dataset name not provided.")
        candidate = "-".join(lmdb_dir_path.parts[-3:])
        if not force:
            dataset_name = input(
                f"Enter dataset name or press enter to use {candidate}:"
            )
        if not dataset_name:
            dataset_name = candidate
        print("Using name:", dataset_name)
        output_file = base_path / f"{dataset_name}.json"

    if output_file.exists():
        print(f"Output file '{output_file}' already exists.")
        if (
            not force
            and "y" not in input("Continue anyway and overwrite? [y/n]: ").lower()
        ):
            exit()

    if not data_mapping_path.exists():
        download_pkl(data_mapping_path, data_mapping_url)

    with data_mapping_path.open("rb") as f:
        data_mapping = pickle.loads(f.read())

    keys, envs = [], []
    for db_path in db_paths:
        envs.append(connect_db(db_path))
        length = envs[-1].begin().get("length".encode("ascii"))
        if length is not None:
            length = pickle.loads(length)
        else:
            length = envs[-1].stat()["entries"]
        assert length is not None, f"Could not find length of LMDB {db_path}"
        keys.append(list(range(length)))

    keylens = [len(k) for k in keys]
    keylen_cumulative = np.cumsum(keylens).tolist()
    num_samples = sum(keylens)

    metadatas = defaultdict(list)

    for i in tqdm(range(num_samples), total=num_samples):
        el_idx, db_idx, sample = get_pickled_from_db(i, keys, envs, keylen_cumulative)
        sid = sample["sid"]
        met = {**data_mapping[f"random{sid}"]}
        met["source"] = dataset_name
        met["ds_idx"] = i
        for k, v in met.items():
            metadatas[k].append(v)

    for env in envs:
        env.close()

    print(f"Saving metadata to {output_file}")
    output_file.write_text(json.dumps(metadatas))
