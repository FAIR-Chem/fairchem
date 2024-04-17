#!/bin/bash

python /home/jovyan/CatTSunami/ocpneb/run_validation/run_validation.py \
    --checkpoint_path "/home/jovyan/checkpoints/eq2_31M_ec4_allmd.pt" \
    --trajectory_path "/home/jovyan/shared-scratch/Brook/neb_stuff/tars/checkpoint/brookwander/neb_results/all_dft_trajs/dft_trajs_for_release/dissociations/" \
    --k 1 \
    --output_file_path "/home/jovyan/trash/" \
    --batch_size 8 \
    --delta_fmax_climb 0.4 \
    --get_ts_sp \
    --vasp_command "mpirun -np 16 --map-by hwthread /opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std" \
    --fmax 0.05  \
    --mapping_file_path "/home/jovyan/CatTSunami/ocpneb/run_validation/mapping_files/dissociation_mapping.pkl" 