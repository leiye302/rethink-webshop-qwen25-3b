import hashlib
import json
from pathlib import Path
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_data_counts_split_and_good_actions():
    spec=json.loads((ROOT/'configs/data.json').read_text())
    assert hashlib.sha256((ROOT/'data/frozen_sft.zip').read_bytes()).hexdigest()==spec['bundle_sha256']
    with zipfile.ZipFile(ROOT/'data/frozen_sft.zip') as z:
        all_rows={}
        for n,expected in spec['files'].items():
            content=z.read(n)
            assert len(content)==expected['bytes']
            assert hashlib.sha256(content).hexdigest()==expected['sha256']
            all_rows[n]=[json.loads(x) for x in content.decode('utf-8-sig').splitlines() if x.strip()]
    for name,g,b in [('train.jsonl',7500,4500),('val.jsonl',631,271),('generation_val200.jsonl',100,100)]:
        rows=all_rows[name]
        assert len(rows)==g+b
        assert sum(r['route']=='good' for r in rows)==g
        assert sum(r['route']=='bad' for r in rows)==b
        assert len({r['audit_id'] for r in rows})==len(rows)
        for r in rows:
            assert r['assistant_prefix'].startswith('<think>')
            assert r['assistant_prefix'].rstrip().endswith('</think>')
            assert not r['assistant_prefix'].endswith('</think>')
            assert r['response']==('<RETHINK>' if r['route']=='bad' else '')+f"<action>{r['supervision_action']}</action>"
            if r['route']=='good':
                assert r['source_executed_action']==r['supervision_action']
    for field in ['audit_id','task_id','task_sha256']:
        assert not {r[field] for r in all_rows['train.jsonl']}&{r[field] for r in all_rows['val.jsonl']}
    assert {r['audit_id'] for r in all_rows['generation_val200.jsonl']} <= {r['audit_id'] for r in all_rows['val.jsonl']}


def test_frozen_loss_files_are_exact():
    spec=json.loads((ROOT/'configs/reference.lock.json').read_text())
    for n,expected in spec['files'].items():
        assert hashlib.sha256((ROOT/'src/sft_reference'/n).read_bytes()).hexdigest()==expected
    assert spec['files']['route_bce/core.py']=='634806f26c2f9a859523ead6707c89eb125c83715153d0dd48c6a204b07c119f'


def test_recipes():
    sft=json.loads((ROOT/'configs/sft.json').read_text())
    assert (sft['total_updates'],sft['global_batch'],sft['good_per_batch'],sft['bad_per_batch'])==(1500,8,5,3)
    assert sft['sft_reference_kl']==0 and sft['route_bce_lambda']==0.1
    cfg=json.loads((ROOT/'configs/opsd.json').read_text())
    assert cfg['algorithm.sdar.gate_beta']==5 and cfg['algorithm.sdar.sdar_coef']==.01
    assert cfg['env.history_length']==2 and cfg['env.max_steps']==15
    assert cfg['data.max_prompt_length']==4096 and cfg['data.max_response_length']==512
    assert cfg['trainer.total_training_steps']==150
    assert cfg['trainer.test_freq']==10 and cfg['trainer.save_freq']==20
    assert cfg['trainer.max_actor_ckpt_to_keep']==2
    assert sft['prepared_world_size']==4
    assert cfg['trainer.n_gpus_per_node']==4 and cfg['trainer.nnodes']==1
    assert cfg['actor_rollout_ref.rollout.tensor_model_parallel_size']==2
    assert cfg['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']*4==16
    assert cfg['actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu']*4==32
    assert cfg['actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu']*4==32


def test_native_fixed_validation():
    import numpy as np
    from rethink_webshop.validation import goal_indices,seed_for_worker
    rng=np.random.RandomState(92)
    expected=np.random.RandomState(0).choice(range(500),128,replace=False)
    assert np.array_equal(goal_indices(rng,range(500),128,False),expected)
    assert np.array_equal(goal_indices(rng,range(500),128,False),expected)
    assert seed_for_worker(1000,127,1,False)==0
    assert seed_for_worker(0,19,8,True)==2


def test_full_vocabulary_bce_shift_and_mask():
    torch=pytest.importorskip('torch')
    from route_bce.core import losses
    torch.manual_seed(2)
    ids=torch.tensor([0,1,2,3,5,6])
    logits=torch.randn(6,9,requires_grad=True)
    result=losses(logits,ids,4,1,5,route_lambda=.1)
    expected=torch.nn.functional.cross_entropy(logits[3:-1],ids[4:])
    p=logits[3].softmax(-1)[5]
    assert torch.allclose(result['suffix_ce'],expected)
    assert torch.allclose(result['route_bce'],-p.log(),atol=1e-6)
    result['loss'].backward()
    assert logits.grad[:3].abs().sum()==0
    assert logits.grad[3].abs().sum()>0 and logits.grad[-1].abs().sum()==0
    good=losses(logits.detach(),torch.tensor([0,1,2,3,4,6]),4,0,5)
    assert torch.allclose(good['route_bce'],-torch.log1p(-p),atol=1e-6)


def test_sampler_each_row_once():
    pytest.importorskip('torch')
    from route_bce.core import batches
    rows=[{'route':'good'} for _ in range(7500)]+[{'route':'bad'} for _ in range(4500)]
    order=batches(rows)
    assert len(order)==1500
    assert sorted(i for b in order for i in b)==list(range(12000))
    assert all(sum(rows[i]['route']=='good' for i in b)==5 for b in order)


def test_teacher_static_and_local_only_wrapping():
    from rethink_webshop.hints import RowSkillProvider
    class Provider:
        def get_privileged_info_from_prompt(self,prompt):
            return 'native static skill'
    plain=RowSkillProvider(Provider(),None)
    assert plain.get_privileged_info_from_prompt('X')=='native static skill'
    local=RowSkillProvider(Provider(),'Use the observed size, not the old size.')
    value=local.get_privileged_info_from_prompt('X')
    assert value.startswith('native static skill\n\n')
    assert 'Use the observed size' in value


def test_resume_never_silently_restarts_from_scratch(tmp_path):
    from rethink_webshop.run import resume_checkpoint
    with pytest.raises(ValueError):
        resume_checkpoint(tmp_path)
    folder=tmp_path/'checkpoints'
    folder.mkdir()
    (folder/'latest_checkpointed_iteration.txt').write_text('20')
    with pytest.raises(ValueError):
        resume_checkpoint(tmp_path)
    checkpoint=folder/'global_step_20'
    (checkpoint/'actor').mkdir(parents=True)
    (checkpoint/'data.pt').write_bytes(b'fixture')
    for rank in range(4):
        for kind in ['model','optim','extra_state']:
            (checkpoint/'actor'/f'{kind}_world_size_4_rank_{rank}.pt').write_bytes(b'fixture')
    assert resume_checkpoint(tmp_path)==checkpoint


def test_old_two_rank_resume_is_rejected(tmp_path):
    from rethink_webshop.run import resume_checkpoint
    folder=tmp_path/'checkpoints'
    checkpoint=folder/'global_step_20'
    (checkpoint/'actor').mkdir(parents=True)
    (folder/'latest_checkpointed_iteration.txt').write_text('20')
    (checkpoint/'data.pt').write_bytes(b'fixture')
    for rank in range(2):
        for kind in ['model','optim','extra_state']:
            (checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
    with pytest.raises(ValueError): resume_checkpoint(tmp_path)
