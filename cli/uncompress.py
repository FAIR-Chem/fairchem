"""
Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
script - preprocess_ef.py
"""

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


def main(ipdir: str, opdir: str, num_workers: int) -> None:
    os.makedirs(opdir, exist_ok=True)

    filelist = glob.glob(os.path.join(ipdir, "*txt.xz")) + glob.glob(
        os.path.join(ipdir, "*extxyz.xz")
    )
    ip_op_pairs: List[Tuple[str, str]] = []
    for filename in filelist:
        fname_base = os.path.basename(filename)
        ip_op_pairs.append((filename, os.path.join(opdir, fname_base[:-3])))

    pool = mp.Pool(num_workers)
    list(
        tqdm(
            pool.imap(decompress_list_of_files, ip_op_pairs),
            total=len(ip_op_pairs),
            desc=f"Uncompressing {ipdir}",
        )
    )
