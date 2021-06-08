#!/bin/bash

## job name
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem-per-cpu=6g
#SBATCH --time=24:00:00
#SBATCH --constraint=pascal
#SBATCH --comment="non-preemtable-cpu-job"
#SBATCH --array=0-39

start=${SLURM_ARRAY_TASK_ID}
BEGIN=0
let start=BEGIN+SLURM_ARRAY_TASK_ID*1
let potend=start+1
endlimit=40

end=$(( potend < endlimit ? potend : endlimit ))

for (( i=${start}; i < ${end}; i++ )); do
	/private/home/mshuaibi/.conda/envs/ocp/bin/python /private/home/mshuaibi/baselines/scripts/preprocess_ef.py --data-path  --out-path data/s2ef/200k/train --num-workers 60 --size 200000 --ref-energy --tags --chunk $i --nodes 40 &
done
wait
