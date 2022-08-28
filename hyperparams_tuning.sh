#!/bin/bash
#SBATCH -J Sweep
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB

module load anaconda/3
conda activate ocp
wandb agent --count 50 mila-ocp/ocp/zrq4hlvi
