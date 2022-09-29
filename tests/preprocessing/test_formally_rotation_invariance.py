from ocpmodels.common.utils import build_config, setup_imports, setup_logging
from minydra import resolved_args
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
import random
import sys
import torch
from ocpmodels.common.transforms import RandomRotate
from ocpmodels.datasets import data_list_collater
from copy import deepcopy


def original_test_rotation_invariance(batch, model, rotation=None):
    """Compare predictions of rotated versions of the same graphs

    Args:
        batch (data.Batch): batch of graphs
        model (data.model): GNN model we test the rotation invariance of
        energy_diff (int, optional): energy difference in predictions across rotated graphs
        rotation (str, optional): type of rotation applied. Defaults to None.

    Returns:
        _type_: metric quantifying the difference in prediction
    """
    random.seed(1)

    # Sampling a random rotation within [-180, 180] for all axes.
    if rotation == "z":
        transform = RandomRotate([-180, 180], [2])
    elif rotation == "x":
        transform = RandomRotate([-180, 180], [0])
    elif rotation == "y":
        transform = RandomRotate([-180, 180], [1])
    else:
        transform = RandomRotate([-180, 180], [0, 1, 2])

    batch_rotated, rot, inv_rot = transform(deepcopy(batch[0]))
    assert not torch.allclose(batch[0].pos, batch_rotated.pos, atol=1e-05)

    # Pass it through the model.
    energies1, _ = model(batch)
    energies2, _ = model([batch_rotated])

    # Compare predicted energies (after inv-rotation).
    print("Perfect invariance:", torch.allclose(energies1, energies2, atol=1e-05))
    energies_diff = torch.abs(energies1 - energies2).sum()

    return energies_diff


if __name__ == "__main__":

    checkpoint = False  # whether to load from a checkpoint

    opts = resolved_args()

    sys.argv[1:] = [
        "--mode=train",
        "--config=configs/is2re/10k/forcenet/new_forcenet.yml",
    ]
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    trainer_config = build_config(args, override_args)

    # Add this to try out on test datasets
    # self.config["test_dataset"] = {'src':'/network/projects/_groups/ocp/oc20/is2re/all/test_id/data.lmdb'}
    # Repeat for test_ood_ads  test_ood_both  test_ood_cat

    setup_imports()
    trainer = registry.get_trainer_class(trainer_config["trainer"])(**trainer_config)

    # Load checkpoint
    if checkpoint == "fa":
        trainer.load_checkpoint(
            checkpoint_path="/network/scratch/a/alexandre.duval/key_checkpoints/2208025_fa_best_checkpoint.pt"
        )
    elif checkpoint == "no_fa":
        trainer.load_checkpoint(
            checkpoint_path="/network/scratch/a/alexandre.duval/key_checkpoints/2207626_sfarinet_nofa_best_checkpoint.pt"
        )
    elif checkpoint == "dpp":
        trainer.load_checkpoint(
            checkpoint_path="/network/scratch/a/alexandre.duval/key_checkpoints/new_dpp_best_checkpoint.pt"
        )
    elif checkpoint == "schnet":
        trainer.load_checkpoint(
            checkpoint_path="/network/scratch/a/alexandre.duval/key_checkpoints/new_schnet_best_checkpoint.pt"
        )
    elif checkpoint == "forcenet":
        trainer.load_checkpoint(
            checkpoint_path="/network/scratch/a/alexandre.duval/key_checkpoints/new_forcenet_best_checkpoint.pt"
        )
    else:
        pass

    # Check for rotation invariance
    trainer.model.eval()
    loader_iter = iter(trainer.val_loader)
    energy_diff = torch.zeros(1, device=trainer.device)
    energy_diff_z = torch.zeros(1, device=trainer.device)

    for i in range(10):
        batch = next(loader_iter)
        energy_diff_z += original_test_rotation_invariance(batch, trainer.model, "z")

    for i in range(10):
        batch = next(loader_iter)
        energy_diff += original_test_rotation_invariance(batch, trainer.model)

    batch_size = len(batch[0].natoms)
    energy_diff_z = energy_diff_z / (i * batch_size)
    energy_diff = energy_diff / (i * batch_size)

    task = registry.get_task_class(trainer_config["mode"])(trainer_config)
    task.setup(trainer)
