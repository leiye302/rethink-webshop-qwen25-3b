"""Execute the unchanged upstream Teacher builder with a CPU-only RPC double."""
import ast
import copy
import json
import sys
import types
import pytest
from rethink_webshop.common import UPSTREAM


def test_teacher_keeps_responses_and_nontriggered_native_context(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    np=pytest.importorskip('numpy')
    if not UPSTREAM.exists():
        pytest.skip('Prepare pinned upstream first')
    class Batch:
        def __init__(self,tensors,non_tensor=None):
            self.batch=tensors; self.non_tensor_batch=non_tensor or {}
        @classmethod
        def from_dict(cls,tensors): return cls(tensors)
        def select_idxs(self,ids):
            return Batch({k:v[ids] for k,v in self.batch.items()},
                         {k:np.array(v,dtype=object)[ids] for k,v in self.non_tensor_batch.items()})
        @classmethod
        def concat(cls,parts):
            return cls({k:torch.cat([p.batch[k] for p in parts]) for k in parts[0].batch})
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,**kw): return [ord(c)+1 for c in text]
        def decode(self,ids,**kw): return ''.join(chr(int(x)-1) for x in ids if x)
    class Provider:
        def get_privileged_info_from_prompt(self,text): return 'native static skill'
    tok=Tokenizer()
    ns={'DataProto':Batch,'SkillProvider':object,'torch':torch,
        'compute_position_id_with_mask':lambda mask:(mask.cumsum(-1)-1).clamp(min=0)}
    tree=ast.parse((UPSTREAM/'verl/trainer/ppo/rlsd_ray_trainer.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_teacher_batch')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged_teacher_builder','exec'),ns)
    builder=ns['build_teacher_batch']
    monkeypatch.setitem(sys.modules,'verl',types.SimpleNamespace(DataProto=Batch))
    monkeypatch.setitem(sys.modules,'verl.trainer.ppo.rlsd_ray_trainer',types.SimpleNamespace(build_teacher_batch=builder))
    from rethink_webshop import hints
    monkeypatch.setattr(hints,'collect_hints',lambda *a,**kw:([None,'local only', 'x'*2000],['not_triggered','applied','applied']))
    response=torch.tensor([[3,4,5],[6,7,8],[9,10,11]])
    prompts=torch.tensor([tok.encode('visible H')]*3)
    full=torch.cat([prompts,response],dim=1)
    student=Batch({'input_ids':full.clone(),'responses':response.clone(),
                   'attention_mask':torch.ones_like(full),'position_ids':torch.zeros_like(full)})
    before=copy.deepcopy(student.batch)
    native=builder(student,Provider(),tok,256)
    class Options(dict):
        __getattr__=dict.__getitem__
    class RPC:
        version=5
        def compute_log_prob(self,batch):
            self.received=batch
            return Batch({'old_log_probs':torch.full((3,3),float(self.version))})
    rpc=RPC()
    trainer=types.SimpleNamespace(config=types.SimpleNamespace(algorithm=types.SimpleNamespace(
         local_hints=Options(audit_dir=str(tmp_path))),data=Options(max_prompt_length=256,truncation='error')),
         skill_provider=Provider(),tokenizer=tok,actor_rollout_wg=rpc,global_steps=1)
    assert torch.equal(hints.score_teacher(trainer,student),torch.full((3,3),5.))
    assert torch.equal(rpc.received.batch['input_ids'][0],native.batch['input_ids'][0])
    assert torch.equal(rpc.received.batch['input_ids'][2],native.batch['input_ids'][2])
    assert not torch.equal(rpc.received.batch['input_ids'][1],native.batch['input_ids'][1])
    assert torch.equal(rpc.received.batch['responses'],response)
    for k in student.batch:
        assert torch.equal(student.batch[k],before[k]),k
    report=json.loads((tmp_path/'update_000001.json').read_text())
    assert report['applied']==1 and report['fallback_reasons']['teacher_prompt_budget']==1
    rpc.version=6
    assert torch.equal(hints.score_teacher(trainer,student),torch.full((3,3),6.))


def test_capture_routes_by_valid_generated_ids_and_keeps_preaction_state():
    torch=pytest.importorskip('torch')
    from rethink_webshop.hints import capture,attach
    class Tokenizer:
        def get_vocab(self): return {'<RETHINK>':7}
        def encode(self,*a,**kw): return [7]
    batch=types.SimpleNamespace(batch={'responses':torch.tensor([[7,1,0],[1,7,0],[7,1,0]]),
             'attention_mask':torch.tensor([[1,1,0],[1,0,0],[1,1,0]])})
    records=[[],[],[]]
    capture(records,Tokenizer(),batch,{'text':['before0','before1','before2']},
        [{'clickables':['before button']}]*3,['<think>R</think><RETHINK><action>A</action>']*3,
        {'text':['after0','after1','after2']},[0,0,0],[False]*3,[True,True,False],0)
    assert records[0][0]['triggered'] and not records[1][0]['triggered']
    assert not records[2]
    assert records[0][0]['visible_prompt']=='before0'
    assert records[0][0]['feedback']=='after0'
    rows=[[{}],[{}],[{}]]
    attach(records,rows)
    assert rows[0][0]['rethink_triggered'] is True
    assert rows[1][0]['rethink_triggered'] is False
