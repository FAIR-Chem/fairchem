import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))


from ocpmodels.common.registry import registry
from ocpmodels.common.utils import make_trainer
from ocpmodels.preprocessing import (
    one_supernode_per_atom_type,
    one_supernode_per_graph,
    remove_tag0_nodes,
)


def set_data_source(trainer, source):
    trainer.config["val_dataset"] = {
        "src": f"/network/projects/_groups/ocp/oc20/is2re/all/{source}/data.lmdb"
    }

    trainer.val_dataset = registry.get_dataset_class(trainer.config["task"]["dataset"])(
        trainer.config["val_dataset"],
        transform=trainer.fa,
        choice_fa=trainer.choice_fa,
    )
    trainer.val_sampler = trainer.get_sampler(
        trainer.val_dataset,
        trainer.config["optim"].get(
            "eval_batch_size", trainer.config["optim"]["batch_size"]
        ),
        shuffle=False,
    )
    trainer.val_loader = trainer.get_dataloader(
        trainer.val_dataset,
        trainer.val_sampler,
    )


def increment(counter, batch, source):
    if isinstance(batch, list):
        b = batch[0]
    else:
        b = batch

    n_edges = b.edge_index.shape[1]
    n_atoms = b.batch.shape[0]

    rm = remove_tag0_nodes(deepcopy(b))
    sg = one_supernode_per_graph(deepcopy(b))
    sa = one_supernode_per_atom_type(deepcopy(b))

    edges_remove = rm.edge_index.shape[1]
    edges_graph = sg.edge_index.shape[1]
    edges_type = sa.edge_index.shape[1]

    atoms_remove = rm.batch.shape[0]
    atoms_graph = sg.batch.shape[0]
    atoms_type = sa.batch.shape[0]

    counter[source]["edges_baseline"] += n_edges
    counter[source]["edges_remove"] += edges_remove
    counter[source]["edges_graph"] += edges_graph
    counter[source]["edges_type"] += edges_type

    counter[source]["atoms_baseline"] += n_atoms
    counter[source]["atoms_remove"] += atoms_remove
    counter[source]["atoms_graph"] += atoms_graph
    counter[source]["atoms_type"] += atoms_type

    return counter


if __name__ == "__main__":
    trainer = make_trainer(
        str_args=["--mode=train", "--config=configs/is2re/all/schnet/new_schnet.yml"],
        overrides={
            "optim": {
                "num_workers": 6,
                "eval_batch_size": 64,
            },
            "logger": "dummy",
        },
    )
    torch.set_grad_enabled(False)
    counter = defaultdict(lambda: defaultdict(int))
    for i, s in enumerate(["val_ood_ads", "val_ood_cat", "val_ood_both", "val_id"]):
        # Update the val. dataset we look at
        set_data_source(trainer, s)
        for batch in tqdm(trainer.val_loader):
            counter = increment(counter, batch, s)

    for batch in tqdm(trainer.train_loader):
        counter = increment(counter, batch, "train")

    for dataset, counts in counter.items():
        print("-" * 80)
        for k, v in counts.items():
            if "baseline" in k:
                print(f"{dataset} {k}: {v}")
            else:
                is_node = "atoms" in k
                ref = counts["atoms_baseline"] if is_node else counts["edges_baseline"]
                v = v / ref * 100
                print(f"{dataset} {k}: {v:.2f}% of baseline")
