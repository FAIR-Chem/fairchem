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

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401
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
            b = batch[0]
            batch_size = max(b.batch).item() + 1

            # ids of sub-surface nodes, per batch
            sub_nodes = [
                torch.where((b.tags == 0) * (b.batch == i))[0]
                for i in range(batch_size)
            ]
            # single tensor of all the sub-surface nodes
            all_sub_nodes = torch.cat(sub_nodes)

            # idem for non-sub-surface nodes
            non_sub_nodes = [
                torch.where((b.tags != 0) * (b.batch == i))[0]
                for i in range(batch_size)
            ]
            all_non_sub_nodes = torch.cat(non_sub_nodes)

            # super node index per batch: they are last in their batch
            new_sn_ids = [
                sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i
                for i in range(batch_size)
            ]

            # number of aggregated nodes into the super node, per batch
            sn_aggregates = torch.tensor([len(s) for s in sub_nodes])
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
            new_atomic_numbers = torch.cat(
                [
                    torch.cat([b.atomic_numbers[non_sub_nodes[i]], torch.tensor([-1])])
                    for i in range(batch_size)
                ]
            )
            # all super nodes have atomic number -1
            assert all([new_atomic_numbers[s].item() == -1 for s in new_sn_ids])
            # position exclude the sub-surface atoms but include an extra super-node
            new_pos = torch.cat(
                [
                    torch.cat([b.pos[non_sub_nodes[i]], sn_pos[i][None, :]])
                    for i in range(batch_size)
                ]
            )
            # idem
            new_force = torch.cat(
                [
                    torch.cat([b.force[non_sub_nodes[i]], sn_force[i][None, :]])
                    for i in range(batch_size)
                ]
            )

            # edge indices per batch
            ei_batch_ids = [
                (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
                for i in range(batch_size)
            ]
            # boolean src node is not sub per batch
            src_is_not_sub = [
                torch.isin(b.edge_index[0][ei_batch_ids[i]], ns)
                for i, ns in enumerate(non_sub_nodes)
            ]
            # boolean target node is not sub per batch
            target_is_not_sub = [
                torch.isin(b.edge_index[1][ei_batch_ids[i]], ns)
                for i, ns in enumerate(non_sub_nodes)
            ]
            # neither the source nor target node is below the surface
            neither_is_sub = [s * t for s, t in zip(src_is_not_sub, target_is_not_sub)]
            # exactly 1 atom (src XOR target) is below the surface
            one_is_sub = [
                (s.to(torch.int) + t.to(torch.int)) == 1
                for s, t in zip(src_is_not_sub, target_is_not_sub)
            ]
            # edges for which neither src nor target are below the surface
            # and shift to account for super nodes
            ei_neither = [
                b.edge_index[:, ei_batch_ids[e]][:, nis] + e
                for e, nis in enumerate(neither_is_sub)
            ]
            # edges for which neither src XOR target is below the surface
            ei_one = [
                b.edge_index[:, ei_batch_ids[i]][:, ois]
                for i, ois in enumerate(one_is_sub)
            ]
            ei_one_to_sn = []
            for e, eio in enumerate(ei_one):
                # shift because of super nodes
                eio_sn = eio.clone()
                num_nodes = b.natoms[e].item() + 1
                mask = torch.zeros(
                    num_nodes, dtype=torch.bool, device=b.edge_index.device
                )
                mask[non_sub_nodes[e] - non_sub_nodes[e].min()] = 1
                assoc = torch.full(
                    (num_nodes,), -1, dtype=torch.long, device=mask.device
                )
                assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
                eio_sn = assoc[eio]
                # locations for which the original edge links to a sub-surface
                # node are re-wired to the batch's super node
                eio_sn[0][torch.isin(eio[0], sub_nodes[e])] = new_sn_ids[e]
                eio_sn[1][torch.isin(eio[1], sub_nodes[e])] = new_sn_ids[e]
                ei_one_to_sn.append(eio_sn)
            # edges for which both nodes are below the surface (dev only)
            ei_both = [
                ((s + t) == 0) for s, t in zip(src_is_not_sub, target_is_not_sub)
            ]

            # per batch: [edges neither | edges one] and shifted by number of
            # preceding super nodes
            ei_shifted_to_sn = torch.cat(
                [
                    torch.cat([ein, eio], -1)
                    for ein, eio in zip(ei_neither, ei_one_to_sn)
                ],
                -1,
            )

            n_total = [e.sum().item() for e in ei_batch_ids]
            n_kept = [e[0].shape[1] + e[1].shape[1] for e in zip(ei_neither, ei_one)]
            n_removed = [t - k for t, k in zip(n_total, n_kept)]
            n_both = [b.sum().item() for b in ei_both]
            ratios = [f"{k / t:.2f}" for k, t in zip(n_kept, n_total)]
            assert all([r == b for r, b in zip(n_removed, n_both)])
            print(f"total edges {n_total} | kept {n_kept} | removed {n_removed}")
            print(f"Ratio kept {ratios}")
            print(
                "Average keep ratio",
                torch.mean(torch.tensor([float(r) for r in ratios])).item(),
            )

            break

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
