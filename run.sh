
python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
       --mode train --config-yml configs/sweeps/dimenetpp_tiny_200k.yml \
       --amp --identifier tmp --run-dir tmp --distributed

