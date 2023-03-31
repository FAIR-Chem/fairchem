from pathlib import Path
import json
import sys
from tqdm import tqdm
from datetime import datetime
from minydra import resolved_args

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.timer import Times  # noqa E402
from ocpmodels.trainers.single_trainer import SingleTrainer  # noqa E402
from ocpmodels.common.utils import (  # noqa E402
    build_config,
    setup_imports,
    move_lmdb_data_to_slurm_tmpdir,
)
from ocpmodels.common.flags import Flags  # noqa E402


def log(b, timers):
    timers[-1]["natoms"] = len(b.pos)
    timers[-1]["nedges"] = len(b.cell_offsets)


def now():
    """
    Get a string describing the current time & date as:
    YYYY-MM-DD_HH-MM-SS

    Returns:
        str: now!
    """
    return str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")


if __name__ == "__main__":
    restart = resolved_args(strict=False).restart_from_dir

    if not restart:
        print("Must specify restart_from_dir=some/path/to/run_dir")
        sys.exit(1)

    if restart[-1] == "/":
        restart = restart[:-1]

    setup_imports()

    parser = Flags().get_parser()
    args = parser.parse_args()
    args.restart_from_dir = restart

    config = build_config(args, [])
    config = move_lmdb_data_to_slurm_tmpdir(config)
    config["logger"] = "dummy"
    config["optim"]["eval_batch_size"] = 1
    config["optim"]["batch_size"] = 1

    trainer = SingleTrainer(**config)
    times = Times(gpu=True)
    timers = []

    for bs in tqdm(trainer.loaders["val_id"]):
        with times.next("forward") as t:
            _ = trainer.model_forward(bs)
        timers.append({"forward": t.times["forward"][-1]})
        log(bs[0], timers)

    out_name = f"timers-{restart.split('/')[-1]}-{now()}.json"
    Path(out_name).write_text(json.dumps(timers, indent=4))

    print("\nTimes saved in", str(Path(out_name).resolve()))
