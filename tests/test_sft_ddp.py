"""Real 4-process Gloo check of the production sample-mean/DDP/checkpoint path."""
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from route_bce import core
from gpu_resident import GpuResidentAdamW
import distributed as parallel
import train


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=torch.nn.Linear(12,12)
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.projection(torch.nn.functional.one_hot(input_ids,12).float()))
    def save_pretrained(self,directory,**kwargs):
        Path(directory).mkdir()
        torch.save(self.state_dict(),Path(directory)/'toy_weights.pt')


class TinyTokenizer:
    def save_pretrained(self,directory):
        (Path(directory)/'tokenizer_config.json').write_text('{"synthetic":true}')


def rows():
    return [{'input_ids':[1,2,3,8 if i>=5 else 4]+[5]*(1+i%3)+[2],
             'suffix_start':3,'route_label':int(i>=5),'audit_id':f'fixture{i}'} for i in range(8)]


def config():
    return {'route_bce_lambda':.1,'gradient_clip':1.,'prepared_world_size':4,
            'optimizer_backend':'native_torch_adamw_cuda_fp32'}


def operation(model,optimizer,order):
    return train.global_batch(model,optimizer,rows(),order,core,8,config(),cuda=False)


def worker(rank,rendezvous,directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(71)
        model=DistributedDataParallel(TinyModel(),broadcast_buffers=False)
        optimizer=GpuResidentAdamW(model.module,lr=2e-3)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer,lambda _:1.)
        first=operation(model,optimizer,list(range(8)));scheduler.step()
        second=operation(model,optimizer,list(reversed(range(8))));scheduler.step()
        optimizer.release_gradients()
        root=Path(directory)
        identity={'config':config(),'code':{},'train_audit_ids':[r['audit_id'] for r in rows()]}
        order=[list(range(8))]*3
        checkpoint=train.save_checkpoint(model,TinyTokenizer(),optimizer,scheduler,order,config(),identity,
                       2,root/'checkpoints',root,emergency=True)
        saved_weights={k:v.detach().clone() for k,v in model.module.state_dict().items()}
        restored=TinyModel();restored.load_state_dict(saved_weights)
        restored=DistributedDataParallel(restored,broadcast_buffers=False)
        opt2=GpuResidentAdamW(restored.module,lr=2e-3)
        sched2=torch.optim.lr_scheduler.LambdaLR(opt2.optimizer,lambda _:1.)
        assert train.restore_checkpoint(checkpoint,opt2,sched2,identity,order)==2
        torch.testing.assert_close(optimizer.optimizer.param_groups[0]['lr'],opt2.optimizer.param_groups[0]['lr'])
        third=operation(restored,opt2,list(range(8)));sched2.step()
        torch.save({'weights':restored.module.state_dict(),'first':first,'second':second,'third':third,
                    'checkpoint_world_size':train.read_json(checkpoint/'manifest.json')['world_size']},
                    root/f'result{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.name=='nt' or not dist.is_gloo_available(),reason='Linux CPU Gloo CI executes 4 real ranks')
def test_four_rank_gradient_update_checkpoint_rng_resume_matches_serial(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(71)
    serial=TinyModel();optimizer=GpuResidentAdamW(serial,lr=2e-3)
    expected=[operation(serial,optimizer,x) for x in [list(range(8)),list(reversed(range(8))),list(range(8))]]
    mp.spawn(worker,args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):
        result=torch.load(tmp_path/f'result{rank}.pt',weights_only=False)
        assert result['checkpoint_world_size']==4
        for i,key in enumerate(['first','second','third']):
            assert result[key]['sample_count']==8 and result[key]['local_sample_count']==2
            assert result[key]['world_size']==4
            for term in ['loss','suffix_ce','route_bce','gradient_norm_before_clip']:
                assert result[key][term]==pytest.approx(expected[i][term],rel=2e-6,abs=2e-7)
        for key,value in serial.state_dict().items():
            torch.testing.assert_close(result['weights'][key],value,rtol=2e-6,atol=2e-7)
    state=torch.load(tmp_path/'checkpoints/checkpoint_000002/training_state.pt',weights_only=False)
    assert state['world_size']==4 and len(state['rank_rng_states'])==4
    assert state['next_batch']==2


def test_world_size_is_part_of_sft_recipe_and_saved_identity():
    root=Path(__file__).resolve().parents[1]
    cfg=train.load_config(root/'configs/sft.json')
    assert cfg['prepared_world_size']==4 and cfg['global_batch']==8
    assert train.DEFAULTS['prepared_world_size']==4
