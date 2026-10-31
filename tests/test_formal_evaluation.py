from types import SimpleNamespace
import numpy as np
import pytest
from rethink_webshop.common import ROOT


def test_formal_sampling_and_api_disabled(tmp_path):
    from rethink_webshop.run import resolved
    from omegaconf import OmegaConf
    cfg = resolved(tmp_path/'model', tmp_path/'output')
    assert cfg.data.val_batch_size == 140
    assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == .4
    assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
    assert cfg.actor_rollout_ref.rollout.val_kwargs.top_p == 1.
    assert cfg.actor_rollout_ref.rollout.val_kwargs.top_k == -1
    assert cfg.env.max_steps == 15 and cfg.env.history_length == 2
    assert cfg.trainer.test_freq == 10
    assert cfg.ray_init.runtime_env.env_vars.RETHINK_RUN_DIR == str(tmp_path/'output')
    source=(ROOT/'src/rethink_webshop/run.py').read_text(encoding='utf-8')
    assert "if not a.baseline and not a.eval_only:" in source
    assert "OmegaConf.update(cfg,'algorithm.local_hints.enabled',False" in source
    assert 'from rethink_webshop.trainer import WebShopTrainer as SkillSDRayTrainer' in (ROOT/'.deps/sdar/verl/trainer/main_sdar.py').read_text()


def test_marker_is_valid_only_at_its_actual_decision_boundary():
    from rethink_webshop.benchmark_metrics import protocol
    assert protocol('<think>R</think><RETHINK><action>search[x]</action>')['strict_protocol']
    assert protocol('<think>R</think><action>search[x]</action>')['strict_protocol']
    assert not protocol('<think>R<RETHINK></think><action>search[x]</action>')['strict_protocol']
    assert not protocol('<think>R</think><action>search[x]</action><RETHINK>')['strict_protocol']


def test_sft_native_tasks_are_the_exact_shared_set(tmp_path):
    from evaluation import fixed_native_tasks, NATIVE_TASKS
    from rethink_webshop.fixed_validation import manifest
    first=fixed_native_tasks(tmp_path/'tasks.json')
    assert first==fixed_native_tasks(tmp_path/'tasks.json')
    assert NATIVE_TASKS==140
    assert first['selection']==manifest()['selection']
    assert first['task_ids']==[r['task_id'] for r in manifest()['selection']]


def test_no_default_entry_uses_legacy128():
    for name in ('src/rethink_webshop/run.py','src/rethink_webshop/eval.py',
                 'src/rethink_webshop/validation.py','src/sft/evaluation.py','scripts/smoke_native.py'):
        assert 'eval128.json' not in (ROOT/name).read_text(encoding='utf-8')


def test_formal_eval_uses_the_runtime_that_contains_vllm():
    script=(ROOT/'scripts/eval.sh').read_text(encoding='utf-8')
    assert '.venv-opsd/bin/python' in script
    assert '.venv-sft/bin/python' not in script

def test_formal_eval_prepare_executes_without_api_or_cuda(tmp_path, monkeypatch):
    from rethink_webshop import run
    from rethink_webshop import setup
    from omegaconf import OmegaConf
    monkeypatch.setattr(setup, 'verify_framework', lambda: None)
    monkeypatch.setattr(run, 'placeholders', lambda: None)
    monkeypatch.setattr(run, 'confined', lambda value: tmp_path/value)
    monkeypatch.delenv('HINT_API_KEY', raising=False)
    run.main(['--eval-only','--prepare-only','--actor','model','--run-dir','eval'])
    cfg=OmegaConf.load(tmp_path/'eval/prepared_config.yaml')
    assert cfg.trainer.val_only
    assert not cfg.algorithm.local_hints.enabled
    assert cfg.data.val_batch_size == 140
    assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == .4


def test_sft_native_worker_verifies_each_task_seed_before_generation(monkeypatch):
    from native_worker import NativeSessionWorker
    from rethink_webshop.fixed_validation import manifest
    from rethink_webshop import task_split
    chosen=manifest()['selection']
    monkeypatch.setattr(task_split,'task_identity',lambda goal:goal['content_id'])
    constructed=[]
    class World:
        def __init__(self,seed):
            constructed.append(seed)
            self.seed=seed
            self.base=SimpleNamespace(close=lambda:None)
            self.goals=[{'content_id':r['task_id'],'instruction_text':'visible '+str(i)} for i,r in enumerate(chosen)]
        def reset(self,index,session):
            return SimpleNamespace(close=lambda:None),None,'visible '+str(index)
        def visible(self,env,view,obs):
            return {'prompt':obs,'observation':obs}
    worker=NativeSessionWorker.__new__(NativeSessionWorker)
    worker.selection={r['task_id']:r for r in chosen}
    worker.world=World(1000)
    worker.first_world_available=True
    worker.new_world=World
    worker.digest=lambda value:str(value)
    worker.sessions={}
    worker.seen=set()
    items=[dict(session_id='session'+str(i),task_id=r['task_id'],
                worker_seed=r['worker_seed'],goal_index=r['goal_index']) for i,r in enumerate(chosen[:2])]
    result=worker.reset(items)
    assert constructed == [1000,1001]
    assert [r['task_id'] for r in result['items']] == [r['task_id'] for r in chosen[:2]]
    bad=dict(items[0],session_id='changed',worker_seed=0)
    with pytest.raises(ValueError,match='shared140'):
        worker.reset([bad])
    assert constructed == [1000,1001]
    worker.close_sessions(['session0','session1'])
    assert not worker.sessions


def test_sft_native_sampling_is_not_inherited_model_nucleus_default():
    import torch
    from evaluation import generate_batch
    captured=[]
    class Tokenizer:
        padding_side='right'
        pad_token_id=0
        def apply_chat_template(self,*args,**kwargs): return [3,4]
        def decode(self,*args,**kwargs): return '<think>R</think><action>search[x]</action>'
    class Model:
        generation_config=SimpleNamespace(eos_token_id=2)
        def generate(self,**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(sequences=torch.tensor([[3,4,6,2]]))
    model=Model()
    tok=Tokenizer()
    generate_batch(model,tok,[('row','visible H')],7,torch.device('cpu'),benchmark=True)
    assert captured[-1]['do_sample']
    assert captured[-1]['temperature']==.4 and captured[-1]['top_p']==1.
    assert captured[-1]['top_k']==0  # HF spelling of unrestricted top-k.
    assert captured[-1]['repetition_penalty']==1. and captured[-1]['max_new_tokens']==512
    generate_batch(model,tok,[('row','visible H')],7,torch.device('cpu'))
    assert captured[-1]['do_sample'] is False  # Separate 200-H diagnostic.

