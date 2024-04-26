"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import make_script_trainer, make_trainer_from_dir
from ocpmodels.common.gfn import FAENetWrapper
from ocpmodels.trainers import SingleTrainer
from ocpmodels.datasets.lmdb_dataset import DeupDataset
from ocpmodels.datasets.data_transforms import get_transforms

if __name__ == "__main__":

    deup_dataset_chkpt = "/network/scratch/a/alexandre.duval/ocp/runs/4657270/deup_dataset"
    model_chkpt = "/network/scratch/a/alexandre.duval/ocp/runs/4648581/checkpoints/best_checkpoint.pt"

    data_config = {
        "default_val": "deup-val_ood_cat-val_ood_ads",
        "deup-train-val_id": {
            "src": deup_dataset_chkpt
        },
        "deup-val_ood_cat-val_ood_ads": {
            "src": deup_dataset_chkpt
        },
        "train": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/train/",
            "normalize_labels": True,
        },
        "val_id": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_id/"
        },
        "val_ood_cat": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_cat/"
        },
        "val_ood_ads": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_ads/"
        },
        "val_ood_both": {
            "src": "/network/scratch/s/schmidtv/ocp/datasets/ocp/is2re/all/val_ood_both/"
        },
    }

    trainer = make_trainer_from_dir(
        model_chkpt,
        mode="continue",
        overrides={
            "is_debug": True,
            "silent": True,
            "cp_data_to_tmpdir": False,
            "config": "depfaenet-deup_is2re-all",
            "deup_dataset.create": False,
            "dataset": data_config,
        },
        silent=True,
    )

    wrapper = FAENetWrapper(
        faenet=trainer.model,
        transform=get_transforms(trainer.config),
        frame_averaging=trainer.config.get("frame_averaging", ""),
        trainer_config=trainer.config,
    )

    wrapper.freeze()
    loaders = trainer.loaders

    data_gen = iter(loaders["deup-train-val_id"])
    batch = next(data_gen)
    preds = wrapper(batch)

    # trainer.config["dataset"].update({
    #     "deup-train-val_id": {
    #         "src": "/network/scratch/s/schmidtv/ocp/runs/3301084/deup_dataset"
    #     },
    #     "deup-val_ood_cat-val_ood_ads": {
    #         "src": "/network/scratch/s/schmidtv/ocp/runs/3301084/deup_dataset"
    #     },
    #     "default_val": "deup-val_ood_cat-val_ood_ads"
    # })

    # deup_dataset_path = "/network/scratch/a/alexandre.duval/ocp/runs/4642835/deup_dataset"
    # deup_dataset = DeupDataset(
    #     {
    #         **trainer.config["dataset"],  
    #     },
    #     "deup-train-val_id",
    #     transform=get_transforms(trainer.config),
    # )

    # deup_sample = deup_dataset[0]