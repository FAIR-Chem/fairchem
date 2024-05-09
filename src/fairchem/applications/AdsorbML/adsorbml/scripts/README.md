
## How to run AdsorbML

**Step 0**: Setup: LMDBs, trajectories, and metadata files should be downloaded.

**Step 1**: Run ML relaxations on the LMDB to generate predictions of relaxed energy and relaxed structures.
For example, see [these instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md#initial-structure-to-relaxed-structure-is2rs). We will be using the trajectory files.

**Step 2**: Process ML relaxed structures (MLRS), group by system, and sort configurations by energy.
```
python process_mlrs.py \
    --ml-trajs-path $MODEL_ROOT_DIR/ml_trajs/ \
    --outdir $MODEL_ROOT_DIR/cache/ \
    --metadata $DATA_ROOT_DIR/mappings/oc20dense_mapping.pkl \
    --surface-dir $DATA_ROOT_DIR/trajs/
```

**Step 3**: Select the best k and write DFT input files (VASP used in this work)
There are different parameters for single points vs. relaxations.
```
SP="--sp --nsw 0 --nelm 300"
RX="--nsw 2000 --nelm 60"
python write_top_k_vasp.py \
    --cache $MODEL_ROOT_DIR/cache/cache_sorted_byE.pkl \
    --outdir $MODEL_ROOT_DIR/dft/ \
    --k 5 $SP
```

**Step 4**: Run DFT single points or relaxations on best k.

Be sure to filter for convergence, see `utils.py` for example code.

**Step 5**: Organize ML and DFT results in the required format outlined [here](https://github.com/Open-Catalyst-Project/AdsorbML/blob/d03b35133e3d21b4a88f44618549bf87a83237a6/scripts/dense_eval.py#L31-L67).

A sample submission file can be downloaded here: https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/gemnet_oc_ml_sp_sample_results.tar. Note - configurations failing the physical constraints were excluded from the submission file, do NOT rely on matching each entry. This is intended to provide you with an idea to the format.

**Step 6**: Evaluate results with `dense_eval.py` with the specified format of predictions to get the success rate and DFT speedup.
