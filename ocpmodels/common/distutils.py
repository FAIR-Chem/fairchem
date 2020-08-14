import subprocess

import torch.distributed as dist
import os


def setup(config):
    if config["submit"]:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config["init_method"] = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config["distributed_port"],
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config["world_size"] % nnodes == 0
                    gpus_per_node = config["world_size"] // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config["rank"] = node_id * gpus_per_node
                    config["local_rank"] = 0
                else:
                    assert ntasks_per_node == config["world_size"] // nnodes
                    config["rank"] = int(os.environ.get("SLURM_PROCID"))
                    config["local_rank"] = int(os.environ.get("SLURM_LOCALID"))

                print("Init: ", config["init_method"], config["world_size"], config["rank"])
                dist.init_process_group(
                    backend=config["distributed_backend"],
                    init_method=config["init_method"],
                    world_size=config["world_size"],
                    rank=config["rank"]
                )
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass
    else:
        dist.init_process_group(backend=config["distributed_backend"], init_method='env://')
    # TODO: SLURM


def cleanup():
    dist.destroy_process_group()


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def is_master():
    return get_rank() == 0


def synchronize():
    if get_world_size() == 1:
        return
    dist.barrier()


def all_reduce_tensor(tensor, op=dist.reduce_op.SUM, group=dist.group.WORLD):
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op, group=group)
    return tensor
