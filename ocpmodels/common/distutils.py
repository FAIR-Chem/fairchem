import torch.distributed as dist


def setup(config):
    dist.init_process_group("nccl", init_method='env://')
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
