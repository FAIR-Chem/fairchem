import torch
from minydra import resolved_args
from pathlib import Path
import os
import sys

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except ImportError:
    pass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.datasets.qm7x import InMemoryQM7X


if __name__ == "__main__":
    # typical use:
    # $ python scripts/make_qm7x_preprocessed.py input_dir=/path/to/qm7x/ output_dir=/path/to/output_dir/ # noqa: E501
    # NB: input_dir should contain the qm7x uncompressed .h5 files
    # NB2: creating InMemoryQM7X will be very slow (~30mins)
    args = resolved_args(
        defaults={
            "input_dir": None,
            "output_dir": None,
            "print_every": 1000,
            "max_structures": -1,
            "set_id": None,
            "set_ids": None,
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
        set_ids = [x.strip() for x in args.set_ids.split(",")]

    qm7x = InMemoryQM7X(
        f"{str(input_dir)}/",
        features=[],
        ignore_features=[],
        y="",
        attribute_remapping={},
        max_structures=args.max_structures,
        selector=[".+"],
        set_ids=set_ids,
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

        if i and i % args.print_every == 0:
            print(f"Processed {i} structures")

    out = output_dir / f"{idmol}.pt"
    torch.save(batch_list, out)
