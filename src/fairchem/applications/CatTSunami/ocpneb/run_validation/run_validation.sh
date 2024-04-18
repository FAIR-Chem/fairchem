#!/bin/bash

python run_validation.py \
    --checkpoint_path "<path-to-checkpoint>" \
    --trajectory_path "<path-to-OC20NEB-trajs>" \
    --k 1 \
    --output_file_path "<path-to-your-output>" \
    --batch_size 8 \
    --delta_fmax_climb 0.4 \
    --get_ts_sp \
    --vasp_command "<vasp-command>" \
    --fmax 0.05  \
    --mapping_file_path "mapping_files/dissociation_mapping.pkl" 