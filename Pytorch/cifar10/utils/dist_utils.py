import torch

def rank0_print(*fargs, **kwargs):
    if (not is_dist()) or torch.distributed.get_rank() == 0:
        print(*fargs, **kwargs)

def is_dist():
    return hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized()
