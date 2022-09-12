from pathlib import Path
import sys

if Path.cwd().name == "scripts":
    sys.path.append("..")

from ocpmodels.common.utils import make_trainer

if __name__ == "__main__":

    trainer_conf_overrides = {
        "optim": {
            "num_workers": 4,
            "batch_size": 64,
        },
        "logger": "dummy",
    }

    trainer = make_trainer(
        str_args=["--mode=train", "--config=configs/is2re/all/schnet/new_schnet.yml"],
        overrides=trainer_conf_overrides,
    )
