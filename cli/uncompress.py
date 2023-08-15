"""
Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
script - preprocess_ef.py
"""

import argparse
import glob
import lzma
import multiprocessing as mp
import os
from typing import List, Tuple

from tqdm import tqdm


def read_lzma(inpfile: str, outfile: str) -> None:
    with open(inpfile, "rb") as f:
        contents = lzma.decompress(f.read())
        with open(outfile, "wb") as op:
            op.write(contents)


def decompress_list_of_files(ip_op_pair: Tuple[str, str]) -> None:
    ip_file, op_file = ip_op_pair
    read_lzma(ip_file, op_file)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ipdir", type=str, help="Path to compressed dataset directory"
    )
    parser.add_argument(
        "--opdir", type=str, help="Directory path to uncompress files to"
    )
    parser.add_argument(
        "--num-workers", type=int, help="# of processes to parallelize across"
    )
    return parser


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.opdir, exist_ok=True)

    filelist = glob.glob(os.path.join(args.ipdir, "*txt.xz")) + glob.glob(
        os.path.join(args.ipdir, "*extxyz.xz")
    )
    ip_op_pairs: List[Tuple[str, str]] = []
    for filename in filelist:
        fname_base = os.path.basename(filename)
        ip_op_pairs.append(
            (filename, os.path.join(args.opdir, fname_base[:-3]))
        )

    pool = mp.Pool(args.num_workers)
    list(
        tqdm(
            pool.imap(decompress_list_of_files, ip_op_pairs),
            total=len(ip_op_pairs),
            desc=f"Uncompressing {args.ipdir}",
        )
    )


if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    main(args)
