
## How to run AdsorbML

**Step 0**: setup
LMDBs, trajectories, and metadata files should be downloaded.

**Step 1**: run ML relaxations on the LMDB to generate predictions of relaxed energy and relaxed structures.
For example, see [these instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md#initial-structure-to-relaxed-structure-is2rs). We will be using the trajectory files.

**Step 2**: process MLRS, group by system, and sort configurations by energy.
```
python process_mlrs.py \
    --ml-trajs-path $MODEL_ROOT_DIR/ml_trajs/ \
    --outdir $MODEL_ROOT_DIR/cache/ \
    --metadata $DATA_ROOT_DIR/mappings/oc20dense_mapping.pkl \
    --surface-dir $DATA_ROOT_DIR/trajs/
```

**Step 3**: select the best k and write VASP input files
There are different parameters for single points vs. relaxations.
```
SP="--sp --nsw 0 --nelm 300"
RX="--nsw 2000 --nelm 60"
python write_top_k_vasp.py \
    --cache $MODEL_ROOT_DIR/cache/cache_sorted_byE.pkl \
    --outdir $MODEL_ROOT_DIR/dft/ \
    --k 5 $SP
```

**Step 4**: run VASP single points or relaxations on best k
Be sure to filter for convergence, see `utils.py` for example code.

**Step 5**: evaluate
Run dense_eval.py with the specified format of predictions to get the success rate and DFT speedup.
