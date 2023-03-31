from time import time
import multiprocessing as mp
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from minydra import resolved_args
from tqdm import tqdm
import pickle

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except ImportError:
    pass

sys.path.append(str(Path(__file__).resolve().parent.parent))


def write(mp_args):
    worker_id, files, output_path = mp_args

    db_path = str(output_path / f"worker_{worker_id}.lmdb")
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for idx, file in enumerate(tqdm(files, position=worker_id)):
        txn = db.begin(write=True)
        samples = torch.load(file)
        for k, s in enumerate(samples):
            txn.put(
                f"{file.stem}-{k}".encode("ascii"),
                pickle.dumps(s, protocol=-1),
            )
        txn.commit()


if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "input_dir": None,
            "output_dir": None,
            "workers": 1,
            "from_pt": False,
        }
    )

    print(
        "This file assumes `make_qm7x_preprocessed.py` has been run",
        "and has produced `.pt` files in `input_dir`.",
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    if args.output_dir is None:
        output_dir = input_dir
        print(f"Using input_dir (str({input_dir})) as output_dir")

    if output_dir.exists():
        print(f"Output directory {output_dir} already exists, will overwrite")
    else:
        output_dir.mkdir(parents=True)

    files = list(input_dir.glob("*.pt"))
    assert len(files) > 0, f"No pt files found in {str(input_dir)}"

    assert args.workers > 0, "Must have at least 1 worker"

    files_per_worker = np.array_split(files, args.workers)
    pool = mp.Pool(args.workers)

    workers_args = [
        (
            worker_id,
            files_per_worker[worker_id],
            output_dir,
        )
        for worker_id in range(args.workers)
    ]

    start = time()
    for j, _ in enumerate(pool.imap(write, workers_args)):
        print("Job {} took {:.2f} seconds".format(j, time() - start))
