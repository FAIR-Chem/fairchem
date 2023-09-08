#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output="/network/scratch/a/alexandre.duval/ocp/runs/output-%j.txt"  # replace: location where you want to store the output of the job

module load anaconda/3 # replace: load anaconda module
conda activate ocp  # replace: conda env name
cd /home/mila/a/alexandre.duval/ocp/ocp # replace: location of the code
python main.py --config=faenet-is2re-10k --test_ri=True --mode='train' --graph_rewiring='remove-tag-0' --optim.max_epochs=5