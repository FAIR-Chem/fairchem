import argparse
import os
import pickle
from collections import defaultdict

from fairchem.data.oc.utils.vasp import write_vasp_input_files
from tqdm import tqdm

"""
Given a cache from process_mlrs.py, generate VASP input files
for the best k (either single points or relaxations), and write
list of paths of systems to be run.
"""

VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 2000,
    "nelm": 60,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": False,
    "lwave": False,
    "ncore": 4,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=str, help="cache_sorted_byE.pkl from process_mlrs.py"
    )
    parser.add_argument("--outdir", type=str)
    parser.add_argument(
        "--k", type=int, help="Number of best configs per system to run"
    )
    parser.add_argument(
        "--sp",
        action="store_true",
        help="Write for single point runs instead of relaxations",
    )
    parser.add_argument(
        "--nsw",
        default=0,
        type=int,
        help="Should be 0 for single points, 2000 for relaxations",
    )
    parser.add_argument(
        "--nelm",
        default=60,
        type=int,
        help="Should be 300 for single points, 60 for relaxations",
    )
    args = parser.parse_args()

    VASP_FLAGS["nsw"] = args.nsw
    VASP_FLAGS["nelm"] = args.nelm

    global outdir
    outdir = args.outdir

    dft_prefix = "relax" if not args.sp else "sp"
    cache = pickle.load(open(args.cache, "rb"))

    paths_list = []
    top_k_cache = defaultdict(list)
    for system in tqdm(cache):
        for adslab in cache[system][: args.k]:
            top_k_cache[system].append(adslab)
            adslab_id = adslab[0]
            atoms = adslab[1]
            adslab_dir = os.path.join(
                args.outdir, dft_prefix, "inputs", system, adslab_id
            )
            os.makedirs(adslab_dir, exist_ok=True)
            write_vasp_input_files(atoms, outdir=adslab_dir, vasp_flags=VASP_FLAGS)
            paths_list.append(f"{adslab_dir}\n")

    with open(
        os.path.join(args.outdir, dft_prefix, "inputs", "paths_list.txt"), "w"
    ) as f:
        f.writelines(paths_list)

    with open(
        os.path.join(os.path.dirname(args.cache), f"top_{args.k}_cache_sorted_byE.pkl"),
        "wb",
    ) as g:
        pickle.dump(top_k_cache, g)
