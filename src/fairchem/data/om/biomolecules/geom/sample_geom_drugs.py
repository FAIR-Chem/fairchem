from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import tqdm


def write_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to where you untarred the rdkit folder i.e. path/to/rdkit_folder",
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to the output directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # rdkit folder downloaded 2024-03-01
    base_path = args.base_path
    out_path = args.out_path
    drugs_file = os.path.join(base_path, "summary_drugs.json")

    with open(drugs_file) as f:
        drugs_summ = json.load(f)

    # set random seed
    random.seed(42)

    train_id_list = []
    val_id_list = []
    id_mol_dict = {}
    missing_drugs = []

    for drug in tqdm.tqdm(drugs_summ.keys()):
        # not all drugs have necessary keys
        try:
            charge = drugs_summ[drug]["charge"]
            pickle_path = drugs_summ[drug]["pickle_path"]
        except KeyError:
            missing_drugs.append(drug)
            continue
        pp = os.path.join(base_path, pickle_path)
        with open(pp, "rb") as f:
            mol_dict = pickle.load(f)
        rand = random.random()
        # generate train ids with 85% probability - ~5% charged
        # sampling happens at the molecule level
        if charge == 0 and rand > 0.15:
            for i, c in enumerate(mol_dict["conformers"]):
                train_id_list.append(c["geom_id"])
                id_mol_dict[c["geom_id"]] = {
                    "pickle_path": drugs_summ[drug]["pickle_path"],
                    "conf_idx": i,
                    "charge": charge,
                    "molecule": drug,
                }
        # if charge != 0 or rand =< 0.15:
        else:
            for i, c in enumerate(mol_dict["conformers"]):
                val_id_list.append(c["geom_id"])
                id_mol_dict[c["geom_id"]] = {
                    "pickle_path": drugs_summ[drug]["pickle_path"],
                    "conf_idx": i,
                    "charge": charge,
                    "molecule": drug,
                }
    # shuffle train list
    for i in range(3):
        random.shuffle(train_id_list)

    # write out train/val ids, id_mol_dict, and missing drugs
    write_pickle(train_id_list, os.path.join(out_path, "train_ids.pkl"))
    write_pickle(val_id_list, os.path.join(out_path, "val_ids.pkl"))
    write_pickle(id_mol_dict, os.path.join(out_path, "id_mol_dict.pkl"))
    write_pickle(missing_drugs, os.path.join(out_path, "missing_drugs.pkl"))


if __name__ == "__main__":
    main()
