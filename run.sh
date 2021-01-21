######### TODOs #########
# 1. Use DataParallel around scatter
# 2. Experiment with models with small #channels, but more layers
#    within a block (larger num_before_skip, num_after_skip, & num_output_layers)
# 3. Can we use multiple GPUs for graph creation?



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


# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs16.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.2gpu.bs16 \
#     --tasks-per-node 1 --submit --amp
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs32.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.4gpu.bs32 \
#     --tasks-per-node 1 --submit --amp
# python main.py --mode train \
#     --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs64.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.8gpu.bs64 \
#     --tasks-per-node 1 --submit --amp
python main.py --mode train \
    --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs16.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.2gpu.bs16x4 \
    --tasks-per-node 4 --submit --amp --distributed
python main.py --mode train \
    --config-yml configs/sweeps/pardimenetpp_42.2M_12k.bs32.yml --run-dir exp/paralleldpp/pardpp --identifier pardpp.42.2M.4gpu.bs32x2 \
    --tasks-per-node 2 --submit --amp --distributed


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


