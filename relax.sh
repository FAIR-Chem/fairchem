#!/bin/bash


/private/home/anuroops/anaconda3/envs/ocp-models/bin/python -u main.py --mode run-relaxations --config-yml configs/relax/dimenet_is.yml \
                 --distributed --num-gpus 8 --checkpoint checkpoint.pt --num-runs 40 \
                 --slurm-partition priority --submit


/private/home/anuroops/anaconda3/envs/ocp-models/bin/python -u main.py --mode run-relaxations --config-yml configs/relax/dimenet_oos_ads.yml \
                 --distributed --num-gpus 8 --checkpoint checkpoint.pt --num-runs 40 \
                 --slurm-partition priority --submit


/private/home/anuroops/anaconda3/envs/ocp-models/bin/python -u main.py --mode run-relaxations --config-yml configs/relax/dimenet_oos_ads_bulk.yml \
                 --distributed --num-gpus 8 --checkpoint checkpoint.pt --num-runs 40 \
                 --slurm-partition priority --submit

/private/home/anuroops/anaconda3/envs/ocp-models/bin/python -u main.py --mode run-relaxations --config-yml configs/relax/dimenet_oos_bulk.yml \
                 --distributed --num-gpus 8 --checkpoint checkpoint.pt --num-runs 40 \
                 --slurm-partition priority --submit
