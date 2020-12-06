
# python -u main.py --mode train --config-yml configs/sweeps/dimenetpp_tiny_200k_emb64.yml \
#        --amp --identifier tmp --run-dir tmp

# python -u main.py --mode train --config-yml configs/sweeps/dimenetpp_tiny_200k_emb64.yml \
#        --identifier tmp --run-dir tmp

# python -u -m torch.distributed.launch --nproc_per_node=8 main.py --mode train \
#        --config-yml configs/sweeps/dimenetpp_23.6M_200k_emb32.yml --identifier tmp --run-dir tmp --distributed

# python -u main.py --mode train --config-yml configs/sweeps/pardimenetpp_23.6M_200k_emb32.yml --identifier tmp --run-dir tmp


python main.py --mode train --config-yml configs/sweeps/dimenetpp_23.6M_200k_emb32.yml \
       --identifier dimenetpp --run-dir dimenetpp_sweep --num-gpus 8 --distributed \
       --sweep-yml configs/sweeps/sweep_model.yml --submit

