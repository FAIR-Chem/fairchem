# `samples.json`

It contains a dictionary where:

* `"structures"` is a list of 2-lists `[idmol, i]` where:
  * `idmol` is an `int` describing the molecule's ID in the QM7X data
  * `i` is an `int` describing the index of the structure in the list of available structures of `idmol`
* `"splits"` is a dict where each key `"train"` `"val_id"` `"val_ood"` `"test"` maps to the list of sample indices in the `"structures"` list that should be in the corresponding set
  * `"val_id"` contains unknown structures of molecules present in the `"train"` split (overlapping `idmol` but non-overlapping `i`)
  * `"val_ood"` contains structures of unknown molecules (non-overlapping `idmol`)
  * `"test"` contains structures listed in SpookyNet (page 11) which are absent from any other split

Reading the dataset:

```python
from pathlib import Path

if __name__ == "__main__":
    sample_mapping_path = Path("configs/models/qm7x-metadata/samples.json")
    samples = json.loads(sample_mapping_path.read_text())
    data = {
        k: [samples["structures"][i] for i in v] for k, v in samples["splits"].items()
    }
    print("\n".join(f"{k:8}: {len(v)}" for k, v in data.items()))
```

Reproducing the dataset:

```python
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)

    train_val_id_ratio = 0.975
    # train   : 4068193
    # val_id  : 104521
    # val_ood : 12423
    # test    : 10100

    sample_mapping_path = Path("configs/models/qm7x-metadata/samples.json")
    all_samples = json.loads(sample_mapping_path.read_text())["structures"]

    # Make sets containing molecule ids for each splits:
    # from SpookyNet:
    test_idmols = set(
        [
            1771,
            1805,
            1824,
            2020,
            2085,
            2117,
            3019,
            3108,
            3190,
            3217,
            3257,
            3329,
            3531,
            4010,
            4181,
            4319,
            4713,
            5174,
            5370,
            5580,
            5891,
            6315,
            6583,
            6809,
            7020,
        ]
    )
    # all idmols
    all_idmols = set(t[0] for t in all_samples)
    # idmols for train, val_id, val_ood
    non_test_idmols = all_idmols - test_idmols
    # choose randomly val_ood
    val_ood_idmols = set(
        np.random.choice(
            list(non_test_idmols),
            replace=False,
            size=len(test_idmols),
        )
    )

    splits = {"train": [], "val_id": [], "val_ood": [], "test": []}

    # assign sample indices based on their idmol
    for idx, s in enumerate(tqdm(all_samples)):
        idmol, _ = s
        if idmol in test_idmols:
            splits["test"].append(idx)
        elif idmol in val_ood_idmols:
            splits["val_ood"].append(idx)
        else:
            if np.random.rand() < train_val_id_ratio:
                splits["train"].append(idx)
            else:
                splits["val_id"].append(idx)

    assert sorted([i for s in splits.values() for i in s]) == list(
        range(len(all_samples))
    )

    sample_mapping_path.write_text(
        json.dumps({
            "structures": all_samples,
            "splits": splits,
        })
    )
```

# `stats.json`

Mean and std for each physicochemical properties in the QM7-X dataset where each property is considered as a flattened list, i.e. stats do not conserve shape

Reproduce with

```bash
$ python ocpmodels/datasets/qm7x.py
```
