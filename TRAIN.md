# Training and evaluating models on OCP datasets

- [Getting Started](#getting-started)
- [OC20](#oc20)
  - [Initial Structure to Relaxed Energy (IS2RE)](#initial-structure-to-relaxed-energy-prediction-is2re)
    - [IS2RE Relaxations](#is2re-relaxations)
  - [Structure to Energy and Forces (S2EF)](#structure-to-energy-and-forces-s2ef)
  - [Initial Structure to Relaxed Structure (IS2RS)](#initial-structure-to-relaxed-structure-is2rs)
  - [Create EvalAI submission files](#create-evalai-oc20-submission-files)
    - [S2EF/IS2RE](#s2efis2re)
    - [IS2RS](#is2rs)
- [OC22](#oc22)
  - [Initial Structure to Total Relaxed Energy (IS2RE-Total)](#initial-structure-to-total-relaxed-energy-is2re-total)
  - [Structure to Total Energy and Forces (S2EF-Total)](#structure-to-total-energy-and-forces-s2ef-total)
  - [Joint Training](#joint-training)
  - [Create EvalAI submission files](#create-evalai-oc22-submission-files)
    - [S2EF-Total/IS2RE-Total](#s2ef-totalis2re-total)

## Getting Started

The [Open Catalyst Project](https://opencatalystproject.org/) consists of three
distinct tasks:
- Initial Structure to Relaxed Energy prediction (IS2RE)
- Structure to Energy and Forces (S2EF)
- Initial Structure to Relaxed Structure (IS2RS)

This document is a tutorial for training and evaluating models
for each of these tasks as well as generating submission files for the
[evaluation server hosted on EvalAI](https://eval.ai/web/challenges/challenge-page/712/overview).

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

If training with multiple GPUs, GPU load balancing may be used to evenly distribute a batch of variable system sizes across GPUs. Load balancing may either balance by number of atoms or number of neighbors. A `metadata.npz` file must be available in the dataset directory to take advantage of this feature. The following command will generate a  `metadata.npz` file and place it in the corresponding directory.
```
python scripts/make_lmdb_sizes.py --data-path data/s2ef/train/2M --num-workers 8
```
Load balancing is activated by default (in atoms mode). To change modes you can specify the following in your config:
```
optim:
  load_balancing: neighbors
```
For more details, refer to https://github.com/Open-Catalyst-Project/ocp/pull/267.

If you have access to a slurm cluster, we use the [submitit](https://github.com/facebookincubator/submitit) package to simplify multi-node distributed training:
```
python main.py --distributed --num-gpus 8 --num-nodes 6 --submit [...]
```

In the rest of this tutorial, we explain how to train models for each task.

# OC20

## Initial Structure to Relaxed Energy prediction (IS2RE)

In the IS2RE tasks, the model takes the initial structure as an input and predicts the structure’s adsorption energy
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

### IS2RE Relaxations

Alternatively, the IS2RE task may be approached by 2 methods as described in our paper:

- Single Model: Relaxed energy predictions are extracted from relaxed structures generated via ML relaxations from a single model.

    1. Train a S2EF model on both energy and forces as described [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/TRAIN.md#structure-to-energy-and-forces-s2ef)
    2. Using the trained S2EF model, run ML relaxations as described [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/TRAIN.md#initial-structure-to-relaxed-structure-is2rs). Ensure `traj_dir` is uniquely specified in the config as to save out the full trajectory. A sample config can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/configs/s2ef/2M/dimenet_plus_plus/dpp_relax.yml). ** Note ** Relaxations on the complete val/test set may take upwards of 8hrs depending on your available hardware.
    3. Prepare a submission file by running the following command:
        ```
        python scripts/make_submission_file.py --id path/to/id/traj_dir \
                --ood-ads path/to/ood_ads/traj_dir --ood-cat path/to/ood_cat/traj_dir \
                --ood-both path/to/ood_both/traj_dir --out-path submission_file.npz --is2re-relaxations
        ```
- Dual Model: Relaxed energy predictions are extracted from relaxed structures generated via ML relaxations from two models - one for running relaxations and one for making energy predictions.
    1. Train two S2EF models, energy-only and force-only.
    2. Using the trained force-only S2EF model, run ML relaxations as described previously. Ensure `traj_dir` is uniquely specified in the config as to save out the full trajectory. **Note** Relaxations on the complete val/test set may take upwards of 8hrs depending on your available hardware.
    3. In order to make predictions via the energy-only model on the generated trajectories, LMDBs must be constructed via the following command:
        ```
        python scripts/preprocess_relaxed.py --id path/to/id/traj_dir \
              --ood-ads path/to/ood_ads/traj_dir --ood-cat path/to/ood_cat/traj_dir \
              --ood-both path/to/ood_both/traj_dir --out-path $DIR --num-workers $NUM_WORKERS
        ```
        Where `$DIR` specifies the directory to save generated LMDBs. A sub-directory will be created for each of the 4 splits in `$DIR`. `$NUM_WORKERS` is the number of data preprocessing cpu workers to be used.
    4. Update your energy-only config to point the test set to the newly generated LMDBs. Using the trained energy-only S2EF model, generate predictions via `--mode predict` (as you would do for the general IS2RE/S2EF case).
    5. Prepare a submission file by running the following command:
        ```
        python scripts/make_submission_file.py --id path/to/id/s2ef_predictions.npz \
                --ood-ads path/to/ood_ads/s2ef_predictions.npz --ood-cat path/to/ood_cat/s2ef_predictions.npz \
                --ood-both path/to/ood_both/s2ef_predictions.npz --out-path submission_file.npz \
                --is2re-relaxations --hybrid
        ```
## Structure to Energy and Forces (S2EF)

In the S2EF task, the model takes the positions of the atoms as input and predicts the adsorption energy and per-atom
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
  maxstep: 0.04
  memory: 50
  damping: 1.0
  alpha: 70.0
  traj_dir: "trajectories" # specify directory you wish to log the entire relaxations, suppress otherwise
```

After training, relaxations can be run by:
```
python main.py --mode run-relaxations --config-yml configs/s2ef/2M/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The relaxed structure positions are stored in `[RESULTS_DIR]/relaxed_positions.npz` and later used to create a submission file to be uploaded to EvalAI. Predicted trajectories are stored in `trajectories` directory for those interested in analyzing the complete relaxation trajectory.

## Create EvalAI OC20 submission files

EvalAI expects results to be structured in a specific format for a submission to be successful. A submission must contain results from the 4 different splits - in distribution (id), out of distribution adsorbate (ood ads), out of distribution catalyst (ood cat), and out of distribution adsorbate and catalyst (ood both). Constructing the submission file for each of the above tasks is as follows:

### S2EF/IS2RE:
1. Run predictions `--mode predict` on all 4 splits, generating `[s2ef/is2re]_predictions.npz` files for each split.
2. Run the following command:
    ```
    python make_submission_file.py --id path/to/id/file.npz --ood-ads path/to/ood_ads/file.npz \
    --ood-cat path/to/ood_cat/file.npz --ood-both path/to/ood_both/file.npz --out-path submission_file.npz
    ```
   Where `file.npz` corresponds to the respective `[s2ef/is2re]_predictions.npz` files generated for the corresponding task. The final submission file will be written to `submission_file.npz` (rename accordingly).
3. Upload `submission_file.npz` to EvalAI.


### IS2RS:
1. Ensure `write_pos: True` is included in your configuration file. Run relaxations `--mode run-relaxations` on all 4 splits, generating `relaxed_positions.npz` files for each split.
2. Run the following command:
    ```
    python make_submission_file.py --id path/to/id/relaxed_positions.npz --ood-ads path/to/ood_ads/relaxed_positions.npz \
    --ood-cat path/to/ood_cat/relaxed_positions.npz --ood-both path/to/ood_both/relaxed_positions.npz --out-path is2rs_submission.npz
    ```
   The final submission file will be written to `is2rs_submission.npz` (rename accordingly).
3. Upload `is2rs_submission.npz` to EvalAI.

# OC22

## Initial Structure to Total Relaxed Energy (IS2RE-Total)

For the IS2RE-Total task, the model takes the initial structure as input and predicts the total DFT energy of the relaxed structure. This task is more general and more challenging than the original OC20 IS2RE task that predicts adsorption energy. To train an OC22 IS2RE-Total model use the `EnergyTrainer` with the `OC22LmdbDataset` by including these lines in your configuration file:

```
trainer: energy # Use the EnergyTrainer

task:
  dataset: oc22_lmdb # Use the OC22LmdbDataset
  ...
```
You can find examples configuration files in [`configs/oc22/is2re`](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs/oc22/is2re).

## Structure to Total Energy and Forces (S2EF-Total)

The S2EF-Total task takes a structure and predicts the total DFT energy and per-atom forces. This differs from the original OC20 S2EF task because it predicts total energy instead of adsorption energy. To train an OC22 S2EF-Total model use the ForcesTrainer with the OC22LmdbDataset by including these lines in your configuration file:

```
trainer: forces  # Use the ForcesTrainer

task:
  dataset: oc22_lmdb # Use the OC22LmdbDataset
  ...
```
You can find examples configuration files in [`configs/oc22/s2ef`](https://github.com/Open-Catalyst-Project/ocp/tree/main/configs/oc22/s2ef).

## Joint Training

Training on OC20 total energies whether independently or jointly with OC22 requires `total_energy: True` and a path to the `oc20_ref` (download link provided below) to be specified in the configuration file. These are necessary to convert OC20 adsorption energies into their corresponding total energies. The following changes in the configuration file capture these changes:

```
task:
  dataset: oc22_lmdb
  ...
  
dataset:
  train:
    src: data/oc20+oc22/s2ef/train
    normalize_labels: False
    total_energy: True
    #download at https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc20_ref.pkl
    oc20_ref: path/to/oc22_ref.pkl
  val:
    src: data/oc22/s2ef/val_id
    total_energy: True
    oc20_ref: path/to/oc22_ref.pkl
```

You can find an example configuration file at [configs/oc22/s2ef/base_joint.yml](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/oc22/s2ef/base_joint.yml)

## Create EvalAI OC22 submission files

EvalAI expects results to be structured in a specific format for a submission to be successful. A submission must contain results from the 2 different splits - in distribution (id) and out of distribution (ood). Construct submission files for the OC22 S2EF-Total/IS2RE-Total tasks as follows:

### S2EF-Total/IS2RE-Total:
1. Run predictions `--mode predict` on both the id and ood splits, generating `[s2ef/is2re]_predictions.npz` files for each split.
2. Run the following command:
    ```
    python make_submission_file.py --dataset OC22 --id path/to/id/file.npz --ood path/to/ood_ads/file.npz --out-path submission_file.npz
    ```
   Where `file.npz` corresponds to the respective `[s2ef/is2re]_predictions.npz` files generated for the corresponding task. The final submission file will be written to `submission_file.npz` (rename accordingly). The `dataset` argument specifies which dataset is being considered — this only needs to be set for OC22 predictions because OC20 is the default.
3. Upload `submission_file.npz` to EvalAI.
