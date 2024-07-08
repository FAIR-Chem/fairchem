from __future__ import annotations

import argparse
import os
import pickle

import tqdm
from rdkit.Chem.rdmolfiles import MolToXYZFile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to where you untarred the rdkit folder id.e. path/to/rdkit_folder",
    )
    parser.add_argument(
        "--parent_path",
        type=str,
        required=True,
        help="Path to the parent directory where structures will be written",
    )
    parser.add_argument(
        "--geom_ids_path",
        type=str,
        required=True,
        help="Path to geom ids pkl to write structures for",
    )
    parser.add_argument(
        "--idmol_path",
        type=str,
        required=True,
        help="Path to id_mol_dict.pkl",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        required=True,
        help="Index in split ids to start writting files",
    )
    parser.add_argument(
        "--stop_idx",
        type=int,
        required=True,
        help="Index in split ids to stop writting files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # rdkit folder downloaded 2024-03-01
    base_path = args.base_path
    parent_path = args.parent_path
    geom_ids_path = args.geom_ids_path
    idmol_path = args.idmol_path
    start_idx = args.start_idx
    stop_idx = args.stop_idx

    # read split ids
    with open(geom_ids_path, "rb") as f:
        geom_ids = pickle.load(f)

    with open(idmol_path, "rb") as f:
        idmol = pickle.load(f)

    outdir = os.path.join(parent_path, f"{start_idx}_{stop_idx}_geom_structures")
    assert not os.path.exists(outdir), f"Directory {outdir} already exists"
    os.makedirs(outdir)

    for id in tqdm.tqdm(geom_ids[start_idx:stop_idx]):
        pp = os.path.join(base_path, idmol[id]["pickle_path"])
        with open(pp, "rb") as f:
            mol_dict = pickle.load(f)
        # get charge, multiplicity, and conformer index
        charge = idmol[id]["charge"]
        # defaults to 1 if not explicitly set
        multiplicity = idmol[id].get("multiplicity", 1)
        conf_idx = idmol[id]["conf_idx"]
        f_name = os.path.join(outdir, f"{id}_{charge}_{multiplicity}.xyz")
        # write xyz file
        MolToXYZFile(mol_dict["conformers"][conf_idx]["rd_mol"], f_name)


if __name__ == "__main__":
    main()
