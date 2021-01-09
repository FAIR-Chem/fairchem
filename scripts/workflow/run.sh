# python -u -m torch.distributed.launch main.py --mode run-relaxations --config-yml configs/s2ef/200k/schnet/schnet.yml \
#     --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-11-04-23-10-schnet_200k/checkpoint.pt \
#     --distributed --debug

# python main.py --mode run-relaxations --config-yml configs/s2ef/200k/schnet/schnet.yml \
#     --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
#     --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-11-04-23-10-schnet_200k/checkpoint.pt \
#     --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

python main.py --mode run-relaxations --config-yml configs/s2ef/20M/dimenet/dimenetpp_1M.yml \
    --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
    --checkpoint /private/home/anuroops/sync/baselines/dimenetpp/20M/checkpoints/2020-10-28-23-09-04-dimenetpp_1M_20M_run0/checkpoint.pt \
    --amp --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

python main.py --mode run-relaxations --config-yml configs/s2ef/20M/schnet/schnet.yml \
    --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
    --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-05-03-02-48-schnet_20M_run0/checkpoint.pt \
    --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit
