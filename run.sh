######### TODOs #########
# 1. Use DataParallel around scatter
# 2. Experiment with models with small #channels, but more layers
#    within a block (larger num_before_skip, num_after_skip, & num_output_layers)
# 3. Can we use multiple GPUs for graph creation?

python -u main.py --mode train --config-yml configs/sweeps/dimenetpp_10.8M_20M.yml --sweep-yml configs/sweeps/sweep_feats.yml \
    --identifier dpp.10.8M.20M --run-dir exp/dpp_sweep/dpp_20M --amp --num-nodes 4 --tasks-per-node 8 --distributed \
    --submit --slurm-timeout 72 --slurm-partition priority


    #   35183317  priority                          pardpp_42_2M_bs12_2gpu_4tsk_2nd  R   17:55:27      2 learnfair[0696,0884]
    #   35182453  priority                           pardpp_42_2M_bs8_2gpu_4tsk_2nd  R   18:43:35      2 learnfair[0837,0878]
    #   35182455  priority                          pardpp_42_2M_bs16_4gpu_2tsk_2nd  R   18:43:35      2 learnfair[1374,1737]
    #   35182249  priority                          pardpp_42_2M_bs16_4gpu_2tsk_2nd  R   18:54:45      2 learnfair[5266-5267]
    #   35182252  priority                                      dpp_42_2M_8tsk_2nds  R   18:54:45      2 learnfair[5114,5122]



# python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
#     --mode train --config-yml configs/sweeps/dimenetpp_big_12k.yml \
#     --run-dir exp/tmp/paralleldpp --identifier dpp --distributed 

# python -u -m torch.distributed.launch --nproc_per_node=1 main.py \
#     --mode train --config-yml configs/sweeps/pardimenetpp_big_12k.yml \
#     --run-dir exp/tmp/paralleldpp --identifier dpp --tasks-per-node 1 --distributed 



# python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --mode train --config-yml configs/sweeps/dimenetpp_44.4M_12k.yml \
#     --run-dir exp/tmp/paralleldpp --identifier dpp --distributed 

# python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --mode train --config-yml configs/sweeps/dimenetpp_10.8M_12k.yml \
#     --run-dir exp/tmp/paralleldpp --identifier dpp --distributed --amp

# python -u -m torch.distributed.launch --nproc_per_node=4 main.py \
#     --mode train --config-yml configs/sweeps/pardimenetpp_10.8M_12k.bs8.yml \
#     --run-dir exp/tmp/paralleldpp --identifier dpp --distributed --tasks-per-node 4  # --amp



# python main.py --mode train --run-dir exp/paralleldpp/dpp \
#     --config-yml configs/sweeps/dimenetpp_10.8M_2M.yml --identifier dpp.10.8M.2M \
#      --num-nodes 2 --tasks-per-node 8 --distributed --amp --submit --slurm-partition priority

# python main.py --mode train --run-dir exp/paralleldpp/pardpp \
#     --config-yml configs/sweeps/pardimenetpp_10.8M_2M_2gpu.yml --identifier pardpp.10.8M.2M.2gpu \
#     --num-nodes 2 --tasks-per-node 4 --distributed --amp --submit --slurm-partition priority
# python main.py --mode train --run-dir exp/paralleldpp/pardpp \
#     --config-yml configs/sweeps/pardimenetpp_10.8M_2M_4gpu.yml --identifier pardpp.10.8M.2M.4gpu \
#     --num-nodes 2 --tasks-per-node 2 --distributed --amp --submit --slurm-partition priority







# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_44.4M_12k.bs4.yml --run-dir exp/paralleldpp --identifier pardpp.44.4M.2gpu.bs4 \
#     --num-gpus 2 --submit
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_44.4M_12k.bs8.yml --run-dir exp/paralleldpp --identifier pardpp.44.4M.4gpu.bs8 \
#     --num-gpus 4 --submit
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_44.4M_12k.bs16.yml --run-dir exp/paralleldpp --identifier pardpp.44.4M.8gpu.bs16 \
#     --num-gpus 8 --submit


# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_44.4M_12k.yml --run-dir exp/paralleldpp --identifier dpp.44.4M.2gpu \
#     --distributed --num-gpus 2 --submit --slurm-partition priority
# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_44.4M_12k.yml --run-dir exp/paralleldpp --identifier dpp.44.4M.4gpu \
#     --distributed --num-gpus 4 --submit --slurm-partition priority
# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_44.4M_12k.yml --run-dir exp/paralleldpp --identifier dpp.44.4M.8gpu \
#     --distributed --num-gpus 8 --submit --slurm-partition priority


# python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_2M.bs8.2gpu.yml \
#     --run-dir exp/paralleldpp/pardpp.2M --identifier pardpp.42.2M.bs8.2gpu.4tsk.2nd --amp --distributed \
#     --tasks-per-node 4 --num-nodes 2 --submit --slurm-partition priority

# python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_2M.bs12.2gpu.yml \
#     --run-dir exp/paralleldpp/pardpp.2M --identifier pardpp.42.2M.bs12.2gpu.4tsk.2nd --amp --distributed \
#     --tasks-per-node 4 --num-nodes 2 --submit --slurm-partition priority

# python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_2M.bs16.4gpu.yml \
#     --run-dir exp/paralleldpp/pardpp.2M --identifier pardpp.42.2M.bs16.4gpu.2tsk.2nd --amp --distributed \
#     --tasks-per-node 2 --num-nodes 2 --submit --slurm-partition priority

# python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_2M.bs32.8gpu.yml \
#     --run-dir exp/paralleldpp/pardpp.2M --identifier pardpp.42.2M.bs32.8gpu.1tsk.2nd --amp --distributed \
#     --tasks-per-node 1 --num-nodes 2 --submit --slurm-partition priority

# python -u main.py --mode train --config-yml configs/sweeps/dpp42.2M/dimenetpp_42.2M_2M.bs4.yml \
#     --run-dir exp/paralleldpp/dpp.2M --identifier dpp.42.2M.8tsk.2nds --amp --distributed \
#     --tasks-per-node 8 --num-nodes 2 --submit --slurm-partition priority




# # for num in 1 2 3 4; do
# for num in 5; do
#     python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_12k.bs8.2gpu.yml --sweep-yml configs/sweeps/dpp42.2M/sweep${num}.yml \
#         --run-dir exp/paralleldpp/pardpp.sweeps/sweep${num} --identifier pardpp.42.2M.bs8.2gpu.4tsk --amp --distributed \
#         --tasks-per-node 4 --submit --slurm-partition priority
#     python main.py --mode train --config-yml configs/sweeps/dpp42.2M/pardimenetpp_42.2M_12k.bs16.4gpu.yml --sweep-yml configs/sweeps/dpp42.2M/sweep${num}.yml \
#         --run-dir exp/paralleldpp/pardpp.sweeps/sweep${num} --identifier pardpp.42.2M.bs16.4gpu.2tsk --amp --distributed \
#         --tasks-per-node 2 --submit --slurm-partition priority
#     python main.py --mode train --config-yml configs/sweeps/dpp42.2M/dimenetpp_42.2M_12k.bs4.yml --sweep-yml configs/sweeps/dpp42.2M/sweep${num}.yml \
#         --run-dir exp/paralleldpp/pardpp.sweeps/sweep${num} --identifier dpp.42.2M.bs4 --amp --distributed \
#         --tasks-per-node 8 --submit --slurm-partition priority
# done




# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs16.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.2gpu.bs16 \
#     --tasks-per-node 1 --submit --amp
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs32.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.4gpu.bs32 \
#     --tasks-per-node 1 --submit --amp
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs64.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.8gpu.bs64 \
#     --tasks-per-node 1 --submit --amp
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs16.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.2gpu.bs16x4 \
#     --tasks-per-node 4 --submit --amp --distributed
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs32.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.4gpu.bs32x2 \
#     --tasks-per-node 2 --submit --amp --distributed


# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs16.gpu2.yml --identifier pardpp.42.2M.bs16.gpu2x4 \
#     --run-dir exp/paralleldpp --distributed --amp --tasks-per-node 4 --submit
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs32.gpu4.yml --identifier pardpp.42.2M.bs32.gpu4x2 \
#     --run-dir exp/paralleldpp --distributed --amp --tasks-per-node 2 --submit
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs64.gpu8.yml --identifier pardpp.42.2M.bs64.gpu8x1 \
#     --run-dir exp/paralleldpp --distributed --amp --tasks-per-node 1 --submit



# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_42.2M_12k.yml --run-dir exp/paralleldpp --identifier dpp.42.2M.2gpu \
#     --distributed --num-gpus 2 --submit --slurm-partition priority

# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_42.2M_12k.yml --run-dir exp/paralleldpp --identifier dpp.42.2M.4gpu \
#     --distributed --num-gpus 4 --submit --slurm-partition priority

# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_42.2M_12k.yml --run-dir exp/paralleldpp --identifier dpp.42.2M.8gpu \
#     --distributed --num-gpus 8 --submit --slurm-partition priority




# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_10.8M_12k.bs8.yml --run-dir exp/paralleldpp --identifier pardpp.10.8M.2gpu.bs8 \
#     --num-gpus 2 --submit

# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_10.8M_12k.bs16.yml --run-dir exp/paralleldpp --identifier pardpp.10.8M.4gpu.bs16 \
#     --num-gpus 4 --submit

# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_10.8M_12k.bs32.yml --run-dir exp/paralleldpp --identifier pardpp.10.8M.8gpu.bs32 \
#     --num-gpus 8 --submit

# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_10.8M_12k.yml --run-dir exp/paralleldpp --identifier dpp.10.8M.2gpu \
#     --distributed --num-gpus 2 --submit --slurm-partition priority

# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_10.8M_12k.yml --run-dir exp/paralleldpp --identifier dpp.10.8M.4gpu \
#     --distributed --num-gpus 4 --submit --slurm-partition priority

# python -u main.py --mode train \
#     --config-yml configs/sweeps/dimenetpp_10.8M_12k.yml --run-dir exp/paralleldpp --identifier dpp.10.8M.8gpu \
#     --distributed --num-gpus 8 --submit --slurm-partition priority








# forcesx_mae: 0.0515, forcesy_mae: 0.0602, forcesz_mae: 0.0645, forces_mae: 0.0587, forces_cos: 0.2368, forces_magnitude: 0.0863, energy_mae: 0.6038, energy_force_within_threshold: 0.0005, loss: 0.8230, epoch: 2.0000███████████████████████████████████████████████████| 450/450 [01:00<00:00,  8.00it/s]






















