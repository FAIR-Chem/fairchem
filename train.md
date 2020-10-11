
## IS2RE

python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode train --config-yml configs/ocp_is2re/schnet.yml --num-gpus 2 --distributed


python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode predict --config-yml configs/ocp_is2re/schnet.yml --num-gpus 2 --distributed \
        --checkpoint checkpoints/2020-10-11-00-06-55/checkpoint.pt


## S2EF

python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode train --config-yml configs/ocp_s2ef/schnet.yml --num-gpus 2 --distributed


python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode predict --config-yml configs/ocp_s2ef/schnet.yml --num-gpus 2 --distributed \
        --checkpoint checkpoints/2020-10-11-00-52-44/checkpoint.pt


## IS2RS

python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode train --config-yml configs/ocp_is2rs/schnet.yml --num-gpus 2 --distributed


python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode run_relaxations --config-yml configs/ocp_is2rs/schnet.yml --num-gpus 2 --distributed \
        --checkpoint checkpoints/2020-10-11-00-52-44/checkpoint.pt
