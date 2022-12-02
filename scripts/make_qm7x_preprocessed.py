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
    args = resolved_args(
        defaults={
            "input_dir": None,
            "output_dir": None,
            "print_every": 1000,
            "max_structures": -1,
        }
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    assert input_dir.exists(), f"{str(input_dir)} does not exist"

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created {str(output_dir)}")
    else:
        print("WARNING: output_dir already exists, files may be overwritten")

    qm7x = InMemoryQM7X(
        f"{str(input_dir)}/",
        features=[],
        ignore_features=[],
        y="",
        attribute_remapping={},
        max_structures=args.max_structures,
        selector=[".+"],
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
