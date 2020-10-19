# OC20 Task Tutorials
The open compute challenge consists of three distinct tasks. This document is a tutorial
for training and evaluating models for each of these tasks as well as generating submission files for EvalAI.

`main.py` serves as the entry point to run any task. This script requires two command line
arguments at a minimum:
* `--mode MODE`: MODE can be `train`, `predict` or `run-relaxations` to train a model, make predictions
using an existing model, or run machine learning based relaxations using an existing model, respectively.
* `--config-yml PATH`: PATH is the path to a YAML configuration file. We use YAML files to supply all
parameters to the script. The `configs` directory contains a number of example config files.

Running `main.py` directly runs the model on a single CPU or GPU if one is available:
```
python main.py --mode train --config-yml configs/TASK/SIZE/MODEL/MODEL.yml
```
If you have multiple
GPUs, you can use distributed data parallel training by running:
```
python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 8 [...]
```
`torch.distributed.launch` launches multiple processes for distributed training. For more details, refer to
https://pytorch.org/docs/stable/distributed.html#launch-utility

If you have access to a slurm cluster, we use the [submitit](https://github.com/facebookincubator/submitit) package to simplify multi-node distributed training:
```
python main.py --distributed --num-gpus 8 --num-nodes 6 --submit [...]
```

In the rest of this tutorial, we explain how to train models for each task.

## Initial Structure to Relaxed Energy prediction (IS2RE)

In the IS2RE tasks, the model takes the initial structure as an input and predicts the structure’s energy
in the relaxed state. To train a model for the IS2RE task, you can use the `EnergyTrainer`
Trainer and `SinglePointLmdb` dataset by specifying the following in your configuration file:
```
trainer: energy # Use the EnergyTrainer

dataset:
  # Train data
  - src: [Path to training data]
    normalize_labels: True
    # Mean and standard deviation of energies
    target_mean: -0.969171404838562
    target_std: 1.3671793937683105
  # Val data (optional)
  - src: [Path to validation data]
  # Test data (optional)
  - src: [Path to test data]
```
You can find examples configuration files in [`configs/is2re`](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs/is2re).

To train a SchNet model for the IS2RE task on the 10k split, run:
```
python main.py --mode train --config-yml configs/is2re/10k/schnet/schnet.yml
```

Training logs are stored in `logs/tensorboard/[TIMESTAMP]` where `[TIMESTAMP]` is
the starting time-stamp of the run. You can monitor the training process by running:
```
tensorboard --logdir logs/tensorboard/[TIMESTAMP]
```
At the end of training, the model checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`.

Next, run this model on the test data:
```
python main.py --mode predict --config-yml configs/is2re/10k/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The predictions are stored in `[RESULTS_DIR]/is2re_predictions.npz` and later used to create a submission file to be uploaded to EvalAI.

## Structure to Energy and Forces (S2EF)

In the S2EF task, the model takes the positions of the atoms as input and predicts the energy and per-atom
forces as calculated by DFT. To train a model for the S2EF task, you can use the `ForcesTrainer` Trainer
and `TrajectoryLmdb` dataset by specifying the following in your configuration file:
```
trainer: forces  # Use the ForcesTrainer

dataset:
  # Training data
  - src: [Path to training data]
    normalize_labels: True
    # Mean and standard deviation of energies
    target_mean: -0.7586356401443481
    target_std: 2.981738567352295
    # Mean and standard deviation of forces
    grad_target_mean: 0.0
    grad_target_std: 2.981738567352295
  # Val data (optional)
  - src: [Path to validation data]
  # Test data (optional)
  - src: [Path to test data]
```
You can find examples configuration files in [`configs/s2ef`](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs/s2ef).

To train a SchNet model for the S2EF task on the 2M split using 2 GPUs, run:
```
python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode train --config-yml configs/s2ef/2M/schnet/schnet.yml --num-gpus 2 --distributed
```
Similar to the IS2RE task, tensorboard logs are stored in `logs/tensorboard/[TIMESTAMP]` and the
checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`.

Next, run this model on the test data:
```
python main.py --mode predict --config-yml configs/s2ef/2M/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The predictions are stored in `[RESULTS_DIR]/s2ef_predictions.npz` and later used to create a submission file to be uploaded to EvalAI.

## Initial Structure to Relaxed Structure (IS2RS)

In the IS2RS task the model takes as input an initial structure and predicts the atomic positions in their
final, relaxed state. This can be done by training a model to predict per-atom forces similar to the S2EF
task and then running an iterative relaxation. Although we present an iterative approach, models that directly predict relaxed states are also possible. The iterative approach IS2RS task uses the same configuration files as the S2EF task `configs/s2ef` and follows the same training scheme above.

To perform an iterative relaxation, ensure the following is added to the configuration files of the models you wish to run relaxations on:
```
# Relaxation options
relax_dataset:
  src: data/is2re/all/val_id/data.lmdb # path to lmdb of systems to be relaxed (uses same lmdbs as is2re)
write_pos: True
relaxation_steps: 300
relax_opt:
  maxstep: 0.04,
  memory: 50,
  damping: 1.0,
  alpha: 70.0,
  traj_dir: "trajectories" # specify directory you wish to log the entire relaxations, suppress otherwise
```

After training, relaxations can be run by:
```
python main.py --mode run-relaxations --config-yml configs/s2ef/2M/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The relaxed structure positions are stored in `[RESULTS_DIR]/relaxed_positions.npz` and later used to create a submission file to be uploaded to EvalAI. Predicted trajectories are stored in `trajectories` directory for those interested in analyzing the complete relaxation trajectory.

## Create EvalAI submission files

EvalAI expects results to be structured in a specific format for a submission to be successful. A submission must contain results from the 4 different splits - in distribution (id), out of distribution adsorbate (ood ads), out of distribution catalyst (ood cat), and out of distribution adsorbate and catalyst (ood both). Constructing the submission file for each of the above tasks is as follows:

### S2EF/IS2RE:
1. Run predictions `--mode predict` on all 4 splits, generating `[s2ef/is2re]_predictions.npz` files for each split.
2. Modify `scripts/make_submission_file.py` with the corresponding paths of the `[s2ef/is2re]_predictions.npz` files and run to generate your final submission file `[s2ef/is2re]_submission.npz` (filename may be modified).
3. Upload `[s2ef/is2re]_submission.npz` to EvalAI.


### IS2RS:
1. Ensure `write_pos: True` is included in your configuration file. Run relaxations `--mode run-relaxations` on all 4 splits, generating `relaxed_positions.npz` files for each split.
2. Modify `scripts/make_submission_file.py` with the corresponding paths of the `relaxed_positions.npz` files and run to generate your final submission file `[is2rs]_submission.npz` (filename may be modified).
3. Upload `is2rs_submission.npz` to EvalAI.
