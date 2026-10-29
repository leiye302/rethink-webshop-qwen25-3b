"""DDP transport only: keep the frozen global sampler and per-row loss."""
from contextlib import nullcontext
from datetime import timedelta
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def primary():
    return rank()==0


def initialize(expected):
    import torch
    if int(os.environ.get('WORLD_SIZE','1'))!=expected:
        raise RuntimeError(f'Use torchrun with exactly {expected} ranks')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))


def barrier():
    if dist.is_initialized(): dist.barrier()


def gather(value):
    if not dist.is_initialized(): return [value]
    values=[None]*world_size()
    dist.all_gather_object(values,value)
    return values


def broadcast(value):
    if not dist.is_initialized(): return value
    values=[value if primary() else None]
    dist.broadcast_object_list(values,src=0)
    return values[0]


def local_indices(indices):
    if len(indices)%world_size():
        raise ValueError('The global batch must divide over all ranks')
    return list(indices[rank()::world_size()])


def accumulation_context(model,last):
    return model.no_sync() if isinstance(model,DistributedDataParallel) and not last else nullcontext()


def unwrap(model):
    return model.module if isinstance(model,DistributedDataParallel) else model


def wrap(model):
    if world_size()==1: return model
    import torch
    local=torch.cuda.current_device()
    return DistributedDataParallel(model,device_ids=[local],output_device=local,
             broadcast_buffers=False,gradient_as_bucket_view=True)
