import torch
from minydra import resolved_args
from pathlib import Path
import os
import sys
import multiprocessing as mp

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except ImportError:
    pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.datasets.qm7x import InMemoryQM7X


def write(mp_args):
    input_dir, output_dir, max_structures, set_ids, worker_id = mp_args
    if worker_id >= 0:
        print("Starting worker", worker_id)

    qm7x = InMemoryQM7X(
        f"{str(input_dir)}/",
        features=[],
        ignore_features=[],
        y="",
        attribute_remapping={},
        max_structures=max_structures,
        selector=[".+"],
        set_ids=set_ids if isinstance(set_ids, list) else [set_ids],
    )

    idmol = None
    batch_list = []

    for i in range(len(qm7x)):
        q = qm7x[i]
        if idmol is None:
            idmol = q.idmol

        if q.idmol == idmol:
            batch_list.append(q)
        else:
            out = output_dir / f"{idmol}.pt"
            torch.save(batch_list, out)
            idmol = q.idmol
            batch_list = [q]

    if batch_list:
        out = output_dir / f"{idmol}.pt"
        torch.save(batch_list, out)

    print(
        ("" if worker_id < 0 else f"Worker {worker_id} ") + f"Processed {i} structures"
    )


if __name__ == "__main__":
    # typical use:
    # $ python scripts/make_qm7x_preprocessed.py \
    #       input_dir=/path/to/qm7x/ \
    #       output_dir=/path/to/output_dir/ \
    #       set_ids=all \
    #       multi_processing=True \
    # NB: input_dir should contain the qm7x uncompressed .h5 files
    # NB2: creating each InMemoryQM7X will be very slow (~1h)
    args = resolved_args(
        defaults={
            "input_dir": None,
            "output_dir": None,
            "max_structures": -1,
            "set_id": None,
            "set_ids": None,
            "multi_processing": False,
        }
    ).pretty_print()

    input_dir = Path(args.input_dir).expanduser().resolve()
    assert input_dir.exists(), f"{str(input_dir)} does not exist"

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created {str(output_dir)}")
    else:
        print("WARNING: output_dir already exists, files may be overwritten")

    set_ids = None
    if args.set_id is not None:
        if args.set_id == "from_procid":
            args.set_id = (int(os.environ["SLURM_PROCID"]) + 1) * 1000
            print("Setting `set_id` to", args.set_id)
        set_ids = [str(int(args.set_id)).strip()]
    if args.set_ids is not None:
        if args.set_ids == "all":
            set_ids = [str(i).strip() for i in range(1000, 9000, 1000)]
        else:
            set_ids = [x.strip() for x in args.set_ids.split(",")]

    if args.multi_processing:
        print("Using multi processing: 1 worker per set_id.")
        if args.set_ids != "all":
            print("Set `set_ids` to `all` to use all 8 set_ids.")
        mp_args = [
            (input_dir, output_dir, args.max_structures, s_id, s)
            for s, s_id in enumerate(set_ids)
        ]
        pool = mp.Pool(processes=len(set_ids))
        list(pool.imap(write, mp_args))
        pool.close()
    else:
        write((input_dir, output_dir, args.max_structures, set_ids, -1))
