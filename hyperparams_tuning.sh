#!/bin/bash
#SBATCH -J Sweep
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output='/network/scratch/a/alexandre.duval/ocp/runs/sweep/output-%j.out'

module load anaconda/3
conda activate ocp
wandb agent --count 40 mila-ocp/ocp/8p8wecso   #ib5jnh2u
