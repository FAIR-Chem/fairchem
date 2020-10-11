
The open compute challenge consists of three distinct tasks. This document is a tutorial 
for training and evaluating models for each of these tasks.

`main.py` serves as the entry point to run any task tasks. This script requires two command line 
arguments at a minimum:
* `--mode MODE`: MODE can be `train`, `predict` or `run_relaxations` to train a model, make predictions 
using an existing model, or run machine learning based relaxations using an existing model respectively.
* `--config-yml PATH`: PATH is the path to a YAML configuration file. We use YAML files to supply all 
parameters to the script. The `configs` directory contains a number of example config files.

Running `main.py` directly runs the model on a single CPU or GPU if one is available. If you have multiple
GPUs, you can use distributed data parallel training by running:
```
python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 8 [...] 
```
`torch.distributed.launch` launches multiple processes for distributed training. For more details, refer to 
https://pytorch.org/docs/stable/distributed.html#launch-utility

If you have access to a slurm cluster, we use the `submitit` package to simplify multi-node distributed training:
```
python main.py --distributed --num-gpus 8 --num-nodes 6 --submit [...] 
```

In the rest of this tutorial, we explain how to train models for each task.

## Initial Structure to Relaxed Energy prediction (IS2RE)

In the IS2RE tasks, the model takes the initial structure as input and predict the structure’s energy
in the relaxed state. To train a model for the IS2RE task, you can use the `EnergyTrainer`
Trainer and `SinglePointLMDB` dataset by specifying the following in your configuration file:
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
You can find examples configuration files in `configs/ocp_is2re`.

To train a SchNet model for the IS2RE task, run:
```
python main.py --mode train --config-yml configs/ocp_is2re/schnet.yml
```

Training logs are stored in `logs/tensorboard/[TIMESTAMP]` where `[TIMESTAMP]` is 
the starting time stamp of the run. You can monitor the training process by running:
```
tensorboard --logdir logs/tensorboard/[TIMESTAMP]
```
At the end of training, the model checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`.

Next, run this model on the test data:
```
python main.py --mode predict --config-yml configs/ocp_is2re/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The predictions are stored in `predictions.txt` which can be uploaded to EvalAI.

## Structure to Energy and Forces (S2EF)

In the S2EF task, the model takes the positions of the atoms as input and predicts the energy and per-atom
forces as calculated by DFT. To train a model for the S2EF task, you can use the `ForcesTrainer` Trainer 
and `SinglePointLMDB` dataset by specifying the following in your configuration file:
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
You can find examples configuration files in `configs/ocp_s2ef`.

To train a SchNet model for the S2EF task, run: 
```
python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode train --config-yml configs/ocp_s2ef/schnet.yml --num-gpus 2 --distributed
```
Similar to the IS2RE task, tensorboard logs are stored in `logs/tensorboard/[TIMESTAMP]` and the 
checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`.

Next, run this model on the test data:
```
python main.py --mode predict --config-yml configs/ocp_s2ef/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The predictions are stored in `predictions.txt` which can be uploaded to EvalAI.

## Initial Structure to Relaxed Structure (IS2RS)

In the IS2RS task the model takes as input an initial structure and predicts the atomic positions in their
final, relaxed state. This can be done by training a model to predict per-atom forces similar to the S2EF
task and then running an iterative relaxation. You can find examples configuration files in `configs/ocp_is2rs`.

To train a SchNet model for the IS2RS task, run: 
```
python main.py --mode train --config-yml configs/ocp_is2rs/schnet.yml
```

After training, you can generate trajectories using:
```
python main.py --mode run_relaxations --config-yml configs/ocp_is2rs/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt
```
The predicted trajectories are stored in `trajectories` directory.
