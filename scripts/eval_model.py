import sys
from copy import deepcopy
from pathlib import Path

from minydra import resolved_args

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import build_config, resolve, setup_imports, merge_dicts
from ocpmodels.trainers.single_trainer import SingleTrainer

if __name__ == "__main__":

    args = resolved_args(
        defaults={
            "job_id": None,
            "dir": None,
            "config": {},
        },
        strict=False,
    )
    assert (
        args.job_id is not None or args.dir is not None
    ), "Must specify either job_id or dir."

    path = (
        resolve(args.dir)
        if args.dir is not None
        else resolve("$SCRATCH/ocp/runs") / str(args.job_id)
    )

    setup_imports()
    argv = deepcopy(sys.argv)
    sys.argv[1:] = []
    trainer_args = flags.parser.parse_args()
    sys.argv[1:] = argv
    trainer_args.continue_from_dir = str(path)
    config = build_config(trainer_args, [])
    config["logger"] = "dummy"
    config["checkpoint"] = str(path / "checkpoints" / "best_checkpoint.pt")
    config = merge_dicts(config, args.config)

    trainer = SingleTrainer(**config)

    trainer.silent = False
    trainer.eval_on_test = True

    trainer.end_of_training(
        -1,
        -1,
        -1,
        [-1],
        from_ckpt=config["checkpoint"],
        disable_tqdm=False,
    )
