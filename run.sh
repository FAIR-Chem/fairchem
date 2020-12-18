
# # python -u -m torch.distributed.launch --nproc_per_node=2 main.py --mode train --config-yml configs/sweeps/forcenet_200k.yml --run-dir tmp --distributed --identifier forcenet.200k

# python -u main.py --mode train --config-yml configs/sweeps/forcenet_200k.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 1 --submit --identifier forcenet.200k 
# python -u main.py --mode train --config-yml configs/sweeps/forcenet_2M.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 2 --submit --identifier forcenet.2M 

# python -u main.py --mode train --config-yml configs/sweeps/forcenet_200k.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 1 --submit --identifier forcenet.200k.amp --amp
# python -u main.py --mode train --config-yml configs/sweeps/forcenet_2M.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 2 --submit --identifier forcenet.2M.amp --amp


# python -u main.py --mode train --config-yml configs/sweeps/forcenet_2M.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 2 --submit --identifier forcenet.2M --sweep-yml configs/sweeps/lr_sweep.yml
# python -u main.py --mode train --config-yml configs/sweeps/forcenet_2M.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 2 --submit --identifier forcenet.2M.amp --amp --sweep-yml configs/sweeps/lr_sweep.yml



# # python -u -m torch.distributed.launch --nproc_per_node=2 main.py --mode train --config-yml configs/sweeps/forcenet_20M_bs8.yml --run-dir tmp --distributed --identifier forcenet.all

# # python -u main.py --mode train --config-yml configs/sweeps/forcenet_20M_bs8.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 4 --submit --identifier forcenet.20M.bs8 
# python -u main.py --mode train --config-yml configs/sweeps/forcenet_20M_bs8_lr0.0001.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 4 --submit --identifier forcenet.20M.bs8.lr0.0001.amp --amp
# python -u main.py --mode train --config-yml configs/sweeps/forcenet_20M_bs16_lr0.0001.yml --run-dir forcenet --distributed --num-gpus 8 --num-nodes 4 --submit --identifier forcenet.20M.bs16.lr0.0001.amp --amp



python -u -m torch.distributed.launch --nproc_per_node=2 main.py --mode run-relaxations --config-yml configs/sweeps/forcenet_2M.yml --run-dir forcenet --checkpoint forcenet/checkpoints/2020-12-16-02-52-48-forcenet.2M_run0/checkpoint.pt --distributed


# LBFGS (Mem = 100)
# {'average_distance_within_threshold': {'total': 268, 'numel': 19600, 'metric': 0.013673469387755101}, 'energy_mae': {'total': 826.4189453125, 'numel': 40, 'metric': 20.6604736328125}}
# Total time taken =  222.94769501686096

# SGD
# {'average_distance_within_threshold': {'total': 0, 'numel': 19600, 'metric': 0.0}, 'energy_mae': {'total': 786.8151245117188, 'numel': 40, 'metric': 19.670378112792967}}
# Total time taken =  172.13674974441528
