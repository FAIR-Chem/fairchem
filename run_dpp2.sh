

# for configfl in configs/eval/dpp_1.8M*; do 
#     python -u main.py --mode predict --config-yml $configfl --run-dir exp/dpp_fn/eval --identifier 2M.`basename $configfl` \
#         --checkpoint FN/dpp_1.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
# done

for configfl in configs/eval/dpp_10.8M*; do 
    python -u main.py --mode predict --config-yml $configfl --run-dir exp/dpp_fn/eval --identifier 2M.`basename $configfl` \
        --checkpoint FN/dpp_10.8M_2M.pt --tasks-per-node 8 --distributed --submit --slurm-partition priority,dev
done

