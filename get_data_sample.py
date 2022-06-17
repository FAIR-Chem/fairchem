"""
Exploration file to get a `batch` in-memory and play around with it.
Use it in notebooks or ipython console

$ ipython
...
In [1]: run get_data_sample.py
Out[1]: ...

In [2]: print(batch)

"""
import sys
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401
from torch import cat, tensor, where, isin
from minydra import resolved_args
from tqdm import tqdm

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports, setup_logging
from ocpmodels.preprocessing import remove_tag0_nodes

if __name__ == "__main__":

    opts = resolved_args()

    sys.argv[1:] = ["--mode=train", "--config=configs/is2re/10k/schnet/schnet.yml"]
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    config["optim"]["num_workers"] = 4
    config["optim"]["batch_size"] = 3
    config["logger"] = "dummy"

    if opts.victor_local:
        config["dataset"][0]["src"] = "data/is2re/10k/train/data.lmdb"
        config["dataset"] = config["dataset"][:1]
        config["optim"]["num_workers"] = 0
        config["optim"]["batch_size"] = opts.bs or config["optim"]["batch_size"]

    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model_attributes=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 100),
        seed=config.get("seed", 0),
        logger=config.get("logger", "wandb"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        new_gnn=config.get("new_gnn", True),
        data_split=config.get("data_split", None),
        note=config.get("note", ""),
    )

    task = registry.get_task_class(config["mode"])(config)
    task.setup(trainer)

    if opts.no_tag_0 is None:
        for batch in trainer.train_loader:
            b = batch[0]
            rewired = remove_tag0_nodes(b)
            break

    if opts.no_single_super_node is None:

        for batch in trainer.train_loader:
            break
        b = batch[0]
        data = deepcopy(b)
        t0 = time()
        batch_size = max(b.batch).item() + 1
        device = b.edge_index.device

        # ids of sub-surface nodes, per batch
        sub_nodes = [
            where((b.tags == 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # single tensor of all the sub-surface nodes
        # all_sub_nodes = torch.cat(sub_nodes)

        # idem for non-sub-surface nodes
        non_sub_nodes = [
            where((b.tags != 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # all_non_sub_nodes = torch.cat(non_sub_nodes)

        # super node index per batch: they are last in their batch
        new_sn_ids = [
            sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i
            for i in range(batch_size)
        ]
        data.ptr = tensor(
            [0] + [nsi + 1 for nsi in new_sn_ids], dtype=b.ptr.dtype, device=device
        )

        # number of aggregated nodes into the super node, per batch
        data.sn_nodes_aggregates = tensor([len(s) for s in sub_nodes], device=device)
        # super node position for a batch is the mean of its aggregates
        sn_pos = [b.pos[sub_nodes[i]].mean(0) for i in range(batch_size)]
        # target relaxed position is the mean of the super node's aggregates
        # (per batch)
        sn_pos_relaxed = [
            b.pos_relaxed[sub_nodes[i]].mean(0) for i in range(batch_size)
        ]
        # the force applied on the super node is the mean of the force applied
        # to its aggregates (per batch)
        sn_force = [b.force[sub_nodes[i]].mean(0) for i in range(batch_size)]

        # per-atom tensors

        # SNs are last in their batch
        data.atomic_numbers = cat(
            [
                cat([b.atomic_numbers[non_sub_nodes[i]], tensor([-1], device=device)])
                for i in range(batch_size)
            ]
        )
        # all super nodes have atomic number -1
        assert all([data.atomic_numbers[s].cpu().item() == -1 for s in new_sn_ids])
        # position exclude the sub-surface atoms but include an extra super-node
        data.pos = cat(
            [
                cat([b.pos[non_sub_nodes[i]], sn_pos[i][None, :]])
                for i in range(batch_size)
            ]
        )
        data.pos_relaxed = cat(
            [
                cat([b.pos_relaxed[non_sub_nodes[i]], sn_pos_relaxed[i][None, :]])
                for i in range(batch_size)
            ]
        )
        # idem
        data.force = cat(
            [
                cat([b.force[non_sub_nodes[i]], sn_force[i][None, :]])
                for i in range(batch_size)
            ]
        )

        # edge indices per batch
        ei_batch_ids = [
            (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
            for i in range(batch_size)
        ]
        # edges per batch
        ei_batch = [b.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]
        # boolean src node is not sub per batch
        src_is_not_sub = [
            isin(b.edge_index[0][ei_batch_ids[i]], ns)
            for i, ns in enumerate(non_sub_nodes)
        ]
        # boolean target node is not sub per batch
        target_is_not_sub = [
            isin(b.edge_index[1][ei_batch_ids[i]], ns)
            for i, ns in enumerate(non_sub_nodes)
        ]
        # edges for which both nodes are below the surface (dev only)
        both_are_sub = [~s & ~t for s, t in zip(src_is_not_sub, target_is_not_sub)]
        # edges for which NOT both nodes are below the surface (dev only)
        not_both_are_sub = [~bas for bas in both_are_sub]
        # ^ turn into [(s|t) for s, t in zip(src_is_not_sub, target_is_not_sub)]
        # when both_are_sub is deleted

        # number of edges that end-up being removed
        data.sn_edges_aggregates = tensor(
            [len(n) - n.sum() for n in not_both_are_sub], device=device
        )

        ei_to_sn = []
        times = []
        assocs = []
        for e, ei in enumerate(ei_batch):
            t = time()
            # future, super-node-adjusted edge index
            ei = ei.clone()[:, not_both_are_sub[e]]
            # number of nodes in this batch: all existing + 1 super node
            num_nodes = b.natoms[e].item() + 1
            # 0-based indices of non-sub-surface nodes in this batch
            batch_non_sub_nodes = non_sub_nodes[e] - b.ptr[e]
            assert (b.tags[(b.batch == e)][batch_non_sub_nodes] > 0).all()
            # 0-based indices of sub-surface nodes in this batch
            batch_sub_nodes = sub_nodes[e] - b.ptr[e]
            # mask to reindex the edges
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=b.edge_index.device)
            # mask is 1 for non-sub nodes
            mask[batch_non_sub_nodes] = 1
            # lookup table
            assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
            # new values for non-sub-indices
            assoc[mask] = torch.arange(
                data.ptr[e], data.ptr[e] + mask.sum(), device=assoc.device
            )
            assocs.append(assocs)
            assert (assoc[batch_sub_nodes] == -1).all()
            assert (assoc[mask] != -1).all()
            # re-index edges ; select only the edges for which not
            # both nodes are sub-surface atoms
            ei_sn = assoc[ei - b.ptr[e]]
            # locations for which the original edge links to a sub-surface
            # node are re-wired to the batch's super node
            assert (ei_sn != new_sn_ids[e]).all()
            ei_sn[isin(ei, sub_nodes[e])] = new_sn_ids[e]
            ei_to_sn.append(ei_sn)
            times.append(time() - t)

        data.edge_index = cat(ei_to_sn, -1).to(dtype=b.edge_index.dtype)
        data.batch = torch.zeros(data.ptr[-1], dtype=b.batch.dtype, device=device)
        for i, p in enumerate(data.ptr[:-1]):
            data.batch[torch.arange(p, data.ptr[i + 1], dtype=torch.long)] = tensor(
                i, dtype=b.batch.dtype
            )

        n_total = [e.sum().item() for e in ei_batch_ids]
        n_kept = [e.shape[-1] for e in ei_to_sn]
        n_removed = [t - k for t, k in zip(n_total, n_kept)]
        n_both = [b.sum().item() for b in both_are_sub]
        ratios = [f"{k / t:.2f}" for k, t in zip(n_kept, n_total)]
        assert all([r == b for r, b in zip(n_removed, n_both)])
        print(f"Total edges {n_total} | kept {n_kept} | removed {n_removed}")
        print(f"Ratios kept {ratios}")
        print(f"Average keep ratio {np.mean([float(r) for r in ratios]):.2f}")
        print(
            "Average ei rewiring processing time",
            f"{np.mean(times):.5f} +/- {np.std(times):.5f}",
        )
        print(
            "Total ei rewiring processing time (batch size",
            f"{batch_size}) {np.sum(times):.3f}",
        )
        tf = time()
        print(f"Total processing time: {tf-t0:.3f}")
        print(f"Total processing time per batch: {(tf-t0) / batch_size:.3f}")

    if opts.plot_tags is not None:
        tags = {
            0: [],
            1: [],
            2: [],
        }
        for batch in tqdm(trainer.train_loader):
            for b in batch:
                for t in tags:
                    tags[t].append((b.tags == t).sum().item())

        x = np.arange(len(tags[0]))
        ys = [np.array(tags[t]) for t in range(3)]
        z = np.zeros(len(x))
        fig = plt.figure(num=1)
        ax = fig.add_subplot(111)
        colors = {
            0: "b",
            1: "y",
            2: "g",
        }
        for t in tags:
            ax.plot(x, ys[t], color=colors[t], lw=1, label=f"tag {t}")
        for t in tags:
            ax.fill_between(
                x, ys[t], where=ys[t] > z, color=colors[t], interpolate=True
            )
        plt.legend()
        plt.savefig("tags_dist.png", dpi=150)
