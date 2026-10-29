"""Explicit single-node 4xA100-80GB contract, shared by both comparison arms."""
WORLD_SIZE = 4
MINIMUM_MEMORY_GIB = 75


def validate_visible_devices(value, expected=WORLD_SIZE):
    devices=[x.strip() for x in (value or '').split(',') if x.strip()]
    if len(devices)!=expected or len(set(devices))!=expected or '-1' in devices:
        raise ValueError(f'Expose exactly {expected} distinct allocated GPU IDs/UUIDs')
    return devices


def validate_properties(devices):
    if len(devices)!=WORLD_SIZE:
        raise RuntimeError('This hardware profile requires four visible GPUs')
    for name,memory in devices:
        if 'A100' not in name or memory<MINIMUM_MEMORY_GIB*2**30:
            raise RuntimeError('Expected four A100 80GB devices, not a 40GB/MIG slice')


def check_cuda(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=WORLD_SIZE:
        raise RuntimeError('Assign exactly four available CUDA devices to this run')
    values=[torch.cuda.get_device_properties(i) for i in range(WORLD_SIZE)]
    validate_properties([(p.name,p.total_memory) for p in values])
    return [{'device':i,'name':p.name,'memory_bytes':p.total_memory} for i,p in enumerate(values)]
