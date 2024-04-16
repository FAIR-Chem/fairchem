#!/bin/bash

#SBATCH --job-name=dft_sps
#SBATCH --output=logs/ml_neb_val_w_dft_sp_test/%A_%a.out
#SBATCH --error=logs/ml_neb_val_w_dft_sp_test/%A_%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --constraint=volta32gb

#SBATCH --array=0-294%32

start=${SLURM_ARRAY_TASK_ID}
BEGIN=0
let start=BEGIN+SLURM_ARRAY_TASK_ID*1
let potend=start+1
endlimit=294

end=$(( potend < endlimit ? potend : endlimit ))

for (( i=${start}; i < ${end}; i++ )); do
    /private/home/brookwander/miniconda3/envs/baseclone/bin/python /private/home/brookwander/ocpneb/ocpneb/workflow/validation/ml_inf_and_get_dft_single_points.py \
        --n_frames 10 \
        --checkpoint_path /private/home/brookwander/ocp_checkpoints/eq2_153M_ec4_allmd.pt \
        --k 1 \
        --output_file_path /checkpoint/brookwander/neb_results/ml_val_w_dft_sp_disoc_202310 \
        --batch_size 2 \
        --delta_fmax_climb 0.4 \
        --fmax 0.05  \
        --get_init_fin_sps False \
        --mapping_file_path /private/home/brookwander/ocpneb/mapping_files/disoc_dft_mapping_202310.pkl \
        --mapping_idx $i 
done
wait