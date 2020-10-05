import torch

from ocpmodels.common.efficient_validation.bfgs_torch import BFGS
from ocpmodels.common.efficient_validation.lbfgs_torch import LBFGS, TorchCalc
from ocpmodels.common.meter import mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry


def relax_eval(
    batch,
    model,
    metric,
    steps,
    fmax,
    relax_opt,
    return_relaxed_pos,
    device="cuda:0",
    transform=None,
):
    """
    Evaluation of ML-based relaxations.
    Args:
        batch: object
        model: object
        metric: str
            Evaluation metric to be used.
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
        return_relaxed_pos: bool
            Whether to return relaxed positions to be written for dft
            evaluation.
    """
    batch = batch[0]
    ids = batch.id
    calc = TorchCalc(model, transform)

    true_relaxed_pos = batch.pos_relaxed
    true_relaxed_energy = batch.y_relaxed

    # Run ML-based relaxation
    if relax_opt["name"] == "bfgs":
        dyn = BFGS(batch, calc)
    elif relax_opt["name"] == "lbfgs":
        traj_dir = relax_opt.get("traj_dir", None)
        dyn = LBFGS(
            batch,
            calc,
            memory=relax_opt["memory"],
            device=device,
            traj_dir=traj_dir,
            traj_names=ids,
        )
    else:
        raise ValueError(f"Unknown relax optimizer: {relax_opt}")

    relaxed_batch = dyn.run(fmax=fmax, steps=steps)

    if return_relaxed_pos:
        natoms = batch.natoms.tolist()
        positions = torch.split(batch.pos, natoms)
        relaxed_positions = [pos.tolist() for pos in positions]
        relaxed_positions = list(zip(ids, relaxed_positions))
    else:
        relaxed_positions = None

    if isinstance(relaxed_batch.y, list):
        ml_relaxed_energy = relaxed_batch.pos.new_tensor(relaxed_batch.y).cpu()
        ml_relaxed_pos = relaxed_batch.pos.cpu()
    else:
        ml_relaxed_energy = relaxed_batch.y.cpu()
        ml_relaxed_pos = relaxed_batch.pos.cpu()

    energy_error = eval(metric)(true_relaxed_energy.cpu(), ml_relaxed_energy)
    structure_error = torch.mean(
        eval(metric)(
            ml_relaxed_pos,
            true_relaxed_pos.cpu(),
        )
    )
    return energy_error, structure_error, relaxed_positions
