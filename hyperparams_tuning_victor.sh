#!/bin/bash
#SBATCH -J Sweep
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output="/network/scratch/s/schmidtv/ocp/runs/sweep/output-%j.out"
module load anaconda/3
conda activate ocp-env
wandb agent --count 2 mila-ocp/ocp/z9y7qle0