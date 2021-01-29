
# python -u main.py --mode train --config-yml configs/sweeps/dpp_base_2.4M_2M.yml \
#     --identifier dpp.2.4M.2M --run-dir exp/dpp_fn/dpp_2M --amp --num-nodes 8 --tasks-per-node 8 --distributed \
#     --submit --slurm-timeout 72 --slurm-partition priority

# python -u main.py --mode train --config-yml configs/sweeps/dpp_base_2.4M_2M.yml \
#     --identifier dpp.2.4M.2M.4nd --run-dir exp/dpp_fn/dpp_2M --amp --num-nodes 4 --tasks-per-node 8 --distributed \
#     --submit --slurm-timeout 72 --slurm-partition priority

# python -u main.py --mode train --config-yml configs/sweeps/dpp_base_2.4M_all.yml \
#     --identifier dpp.2.4M.all.8nd --run-dir exp/dpp_fn/dpp_all --amp --num-nodes 8 --tasks-per-node 8 --distributed \
#     --submit --slurm-timeout 72 --slurm-partition priority

# python -u main.py --mode train --config-yml configs/sweeps/dpp_base_1.8M_2M.yml \
#     --identifier dpp.1.8M.2M --run-dir exp/dpp_fn/dpp_2M --amp --num-nodes 4 --tasks-per-node 8 --distributed \
#     --submit --slurm-timeout 72 --slurm-partition priority



# 2M on ALL
# /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-11-08-18-31-31-dpp1.8M_forceonly_all_restart_ep4.5/checkpoint.pt
# 10.8M on ALL
# /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-11-16-23-34-24-dpp10.8M_forceonly_all_restart_ep2/checkpoint.pt

# 2M on 2M

# 10.8M on 2M
# exp/paralleldpp/dpp/checkpoints/2021-01-20-17-21-04-dpp.10.8M.2M/checkpoint.pt







# for configfl in configs/eval/dpp_1.8M*; do 
#     python -u -m torch.distributed.launch --nproc_per_node=8 \
#         main.py --mode predict --config-yml $configfl --run-dir exp/tmp/eval --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_1.8M_2M.pt --tasks-per-node 8 --distributed
# done


# for configfl in configs/eval/dpp_1.8M*; do 
#     python -u main.py --mode predict --config-yml $configfl --run-dir exp/dpp_fn/eval --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_1.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done

# for configfl in configs/eval/dpp_10.8M*; do 
# for configfl in configs/eval/dpp_10.8M_eval_id.yml configs/eval/dpp_10.8M_eval_ood_both.yml configs/eval/dpp_10.8M_eval_ood_ads.yml; do 
#     python -u main.py --mode predict --config-yml $configfl --run-dir exp/dpp_fn/eval --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_10.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done



# for configfl in configs/eval/dpp_1.8M*ood_both*; do 
#     python -u main.py --mode run-relaxations --config-yml $configfl --run-dir exp/dpp_fn/relax --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_1.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done


# for configfl in configs/eval/dpp_10.8M*ood_both*; do 
#     python -u main.py --mode run-relaxations --config-yml $configfl --run-dir exp/dpp_fn/relax --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_10.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done



# for configfl in configs/eval/dpp_1.8M*; do 
#     python -u main.py --mode run-relaxations --config-yml $configfl --run-dir exp/dpp_fn/relax --identifier all.`basename $configfl` \
#         --checkpoint FN/dpp_1.8M_all.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done

# for configfl in configs/eval/dpp_10.8M*; do 
#     python -u main.py --mode run-relaxations --config-yml $configfl --run-dir exp/dpp_fn/relax --identifier all.`basename $configfl` \
#         --checkpoint FN/dpp_10.8M_all.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done
