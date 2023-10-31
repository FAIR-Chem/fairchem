#!/bin/bash
#SBATCH --job-name=${job-name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --output=${output-dir}/runs/%j/output-main.txt
#SBATCH --tmp=800GB

# Note: replace ${value} by whatever you want to use

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master port $MASTER_PORT"

cd ${code-loc}

module load ${modules}

if ${venv}
then
    source ${venv-loc}/bin/activate
else
    conda activate ${conda-env}
fi

srun --gpus-per-task=1 --output=${output-dir}/runs/%j/output-%t.txt python main.py --config='gemnet_oc-is2re-all' --distributed --num-nodes 1 --num-gpus 4 --test_ri=True --mode='train' --graph_rewiring='remove-tag-0' --model.pg_hidden_channels=32 --optim.max_epochs=20 --optim.batch_size=32 --cp_data_to_tmpdir=False --logdir=${output-dir}/runs/$SLURM_JOB_ID --run-dir=${output-dir}/runs/$SLURM_JOB_ID
