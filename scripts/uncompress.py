"""
Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
script - preprocess_ef.py
"""

import argparse
import glob
import lzma
import multiprocessing
import os


def read_lzma(inpfile, outfile):
    with open(inpfile, "rb") as f:
        contents = lzma.decompress(f.read())
        with open(outfile, "wb") as op:
            op.write(contents)


def decompress_list_of_files(indices, ip_op_pairs):
    for index in indices:
        ip_file, op_file = ip_op_pairs[index]
        read_lzma(ip_file, op_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ipdir", type=str, help="Path to compressed dataset directory"
    )
    parser.add_argument(
        "--opdir", type=str, help="Directory to uncompress files to"
    )
    args = parser.parse_args()

    os.makedirs(args.opdir, exist_ok=True)

    filelist = glob.glob(os.path.join(args.ipdir, "*txt.xz")) + glob.glob(
        os.path.join(args.ipdir, "*extxyz.xz")
    )
    ip_op_pairs = []
    for i in filelist:
        fname_base = os.path.basename(i)
        ip_op_pairs.append((i, os.path.join(args.opdir, fname_base[:-3])))
    k = multiprocessing.cpu_count()
    indices = [i for i in range(len(ip_op_pairs))]
    tasks = [indices[i::k] for i in range(k)]
    procs = []
    # instantiating processes
    for t in tasks:
        proc = multiprocessing.Process(
            target=decompress_list_of_files, args=(t, ip_op_pairs)
        )
        procs.append(proc)
        proc.start()
    # complete the processes
    for proc in procs:
        proc.join()
