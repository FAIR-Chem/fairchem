import sys
import os
import pickle
import copy
import random
import glob
import ase.io
import numpy as np
import torch
from ase.optimize import BFGS, LBFGS
from torch import nn

from bfgs_torch import BFGS as BFGS_torch
from ocpmodels.trainers import ForcesTrainer
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.common.ase_utils import OCPCalculator as OCP
from lbfgs_torch import TorchCalc
from lbfgs_torch import LBFGS as LBFGS_torch
import ase.io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--relaxopt', choices=['bfgs', 'lbfgs'], default='bfgs')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lbfgs-mem', type=int, default=50)
parser.add_argument('--steps', type=int, default=300)
args = parser.parse_args()

task = {
    "dataset": "trajectory_lmdb",
    "description": "Regressing to energies and forces for a trajectory dataset",
    "labels": ["potential energy"],
    "metric": "mae",
    "type": "regression",
    "grad_input": "atomic forces",
    "relax_dataset": {
        "src": "/home/mshuaibi/Documents/ocp-baselines/data/ocp_is2re_1k/"
    },
    "relaxation_steps": args.steps,
}

model = {
    "name": "schnet",
    "hidden_channels": 1024,
    "num_filters": 256,
    "num_interactions": 3,
    "num_gaussians": 200,
    "cutoff": 6.0,
    "use_pbc": True,
}

train_dataset = {
    "src": "./train/",
    "normalize_labels": False,
}

optimizer = {
    "batch_size": 32,
    "eval_batch_size": args.batch_size,
    "lr_gamma": 0.1,
    "lr_initial": 0.0003,
    "lr_milestones": [20, 30],
    "num_workers": 10,
    "max_epochs": 50,
    "warmup_epochs": 10,
    "warmup_factor": 0.2,
    "force_coefficient": 30,
    "criterion": nn.L1Loss(),
    "relax_opt": args.relaxopt,
    "lbfgs_mem": args.lbfgs_mem,
}

identifier = "debug"
trainer = ForcesTrainer(
    task=task,
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    identifier=identifier,
    print_every=5,
    is_debug=True,
    seed=1,
    relax_opt=optimizer["relax_opt"],
    lbfgs_mem=optimizer["lbfgs_mem"],
)

trainer.load_pretrained("./checkpoint.pt")

ref = pickle.load(
    open("/home/mshuaibi/Documents/ocp-baselines/mappings/adslab_ref_energies_full.pkl", "rb")
    # open("/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adslab_ref_energies_full.pkl", "rb")
)

paths = glob.glob("./debug_data/*.traj")
ase_maes = []
torch_maes = []
for traj in paths[:10]:
    images = ase.io.read(traj, ":")
    a2g = AtomsToGraphs(
        max_neigh=50, radius=6, r_energy=True, r_forces=False, r_distances=False
    )
    ref_energy = ref[os.path.basename(traj)[:-5]]
    data_list = a2g.convert_all(images)
    di = data_list_collater([data_list[0]])
    di.pos_relaxed = data_list[-1].pos
    di.y_relaxed = data_list[-1].y - ref[os.path.basename(traj)[:-5]]
    di.y_init = data_list[0].y - ref[os.path.basename(traj)[:-5]]
    del di.y

    # Torch-ML Relaxation 
    use_pbc_graph = True
    model = TorchCalc(trainer, pbc_graph=use_pbc_graph)
    dyn = LBFGS_torch(di, model)
    ml_relaxed = dyn.run(fmax=0, steps=args.steps)
    ml_relaxed_energy = ml_relaxed.y.cpu()
    ml_relaxed_pos = ml_relaxed.pos.cpu()
    mae_torch = torch.abs(ml_relaxed_energy - di.y_relaxed)

    # ASE-ML Relaxation
    calc = OCP(trainer, pbc_graph=use_pbc_graph)
    starting_image = images[0].copy()
    starting_image.set_calculator(calc)
    starting_image.get_potential_energy()
    relaxed = images[-1]
    dyn = LBFGS(starting_image, trajectory="ml_{}".format(os.path.basename(traj)))
    dyn.run(steps=args.steps, fmax=0)
    ml_traj = ase.io.read("ml_{}".format(os.path.basename(traj)), ":")
    dft_energy = relaxed.get_potential_energy()
    ml_energy = ml_traj[-1].get_potential_energy() + ref[os.path.basename(traj)[:-5]]
    mae = np.abs(dft_energy - ml_energy)

    ase_maes.append(mae)
    torch_maes.append(mae_torch)

# Batch - Torch LBFGS
data_list = []
for traj in paths[:10]:
    images = ase.io.read(traj, ":")
    a2g = AtomsToGraphs(
        max_neigh=50, radius=6, r_energy=True, r_forces=False, r_distances=False
    )
    ref_energy = ref[os.path.basename(traj)[:-5]]
    images_list = a2g.convert_all(images)
    di = images_list[0]
    di.pos_relaxed = images_list[-1].pos
    di.y_relaxed = images_list[-1].y - ref[os.path.basename(traj)[:-5]]
    di.y_init = images_list[0].y - ref[os.path.basename(traj)[:-5]]
    del di.y
    data_list.append(di)
batch = data_list_collater(data_list)

use_pbc_graph = True
model = TorchCalc(trainer, pbc_graph=use_pbc_graph)
dyn = LBFGS_torch(batch, model)
ml_relaxed = dyn.run(fmax=0, steps=args.steps)
ml_relaxed_energy = ml_relaxed.y.cpu()
ml_relaxed_pos = ml_relaxed.pos.cpu()
mae_torch = torch.mean(torch.abs(ml_relaxed_energy - batch.y_relaxed.cpu()))
print("batch - 1: ase", np.mean(ase_maes))
print("batch - 1: torch", np.mean(torch_maes))
print("batch - 10: torch", mae_torch)
