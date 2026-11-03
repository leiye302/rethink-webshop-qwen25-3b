"""CPU receipt fixtures: stage selection and dataset-bound Stage1 -> Stage2 handoff."""
import json
from pathlib import Path
import shutil
import pytest
from rethink_webshop import common, stages


def test_method_defaults_to_retrained_local_sft_not_old_release():
    assert stages.default_actor()=='runs/sft/checkpoint_001500/model'
    assert stages.default_actor(True)=='assets/base'
    assert common.load('model.json')['model_id']=='Qwen/Qwen2.5-3B-Instruct'
    assert common.load('data.json')['dataset_revision']=='buy-v9'


@pytest.fixture
def handoff(tmp_path,monkeypatch):
    original=common.ROOT
    (tmp_path/'configs').mkdir()
    for name in ('sft.json','data.json','model.json','model.lock.json'):
        shutil.copyfile(original/'configs'/name,tmp_path/'configs'/name)
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(stages,'ROOT',tmp_path)
    recipe,data,model=common.load('sft.json'),common.load('data.json'),common.load('model.json')
    checkpoint=tmp_path/'runs/sft/checkpoint_001500'
    actor=checkpoint/'model';actor.mkdir(parents=True)
    (actor/'config.json').write_text('{"fixture":true}')
    (actor/'tokenizer.json').write_text('{"fixture":true}')
    (actor/'model.safetensors').write_bytes(b'not-a-real-model; metadata-only test')
    identity={'config':recipe,'config_sha256':stages._digest(recipe),'data':{'files':data['files']},
        'parent':{'revision':model['revision'],'model_lock_sha256':common.sha(tmp_path/'configs/model.lock.json')}}
    manifest={'schema':'webshop-immutable-full-checkpoint-v1','complete':True,
        'completed_updates':1500,'world_size':4}

    def seal():
        (checkpoint/'identity.json').write_text(json.dumps(identity),encoding='utf-8')
        (checkpoint/'config.json').write_text(json.dumps(identity['config']),encoding='utf-8')
        manifest['identity_sha256']=stages._digest(identity)
        manifest['files']={p.relative_to(checkpoint).as_posix():
            {'bytes':p.stat().st_size,'sha256':common.sha(p)}
            for p in checkpoint.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (checkpoint/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    seal()
    return actor,identity,manifest,seal


def test_complete_selected_data_handoff_passes_without_optimizer_or_cuda(handoff):
    actor,identity,manifest,seal=handoff
    result=stages.verify_sft_handoff(actor)
    assert result['completed_updates']==1500 and result['dataset_revision']=='buy-v9'
    assert result['train_sha256']=='860a7b0ac73f91396089365947570c2f3052b2318d553ec1ac204964e78b9e59'
    assert not (actor.parent/'training_state.pt').exists()  # RL initialization is not SFT optimizer resume.


@pytest.mark.parametrize('failure,match',[
    ('old_data','Stage1 data differs'),('wrong_parent','parent model differs'),
    ('intermediate','finish all updates'),('partial','finish all updates'),
    ('world_size','matching-world-size'),('recipe','configuration/identity')])
def test_old_or_incomplete_sft_cannot_initialize_new_method(handoff,failure,match):
    actor,identity,manifest,seal=handoff
    if failure=='old_data':identity['data']['files']['train.jsonl']['sha256']='0'*64
    elif failure=='wrong_parent':identity['parent']['revision']='1.5b-fixture'
    elif failure=='intermediate':manifest['completed_updates']=750
    elif failure=='partial':manifest['complete']=False
    elif failure=='world_size':manifest['world_size']=2
    elif failure=='recipe':
        identity['config']['learning_rate']=1e-4
        identity['config_sha256']=stages._digest(identity['config'])
    seal()
    with pytest.raises(ValueError,match=match):stages.verify_sft_handoff(actor)


def test_tampered_model_or_metadata_is_rejected(handoff):
    actor,identity,manifest,seal=handoff
    target=actor/'model.safetensors';payload=target.read_bytes()
    target.write_bytes(b'x'*len(payload))
    with pytest.raises(ValueError,match='SHA/size'):stages.verify_sft_handoff(actor)
    target.write_bytes(payload)
    (actor.parent/'identity.json').write_text('{}')
    with pytest.raises(ValueError,match='configuration/identity'):stages.verify_sft_handoff(actor)


def test_already_verified_weight_hashes_are_reused(handoff,monkeypatch):
    actor,identity,manifest,seal=handoff
    weight=actor/'model.safetensors';checksum=common.sha(weight)
    original=stages.sha
    def checked(path):
        assert Path(path)!=weight,'Do not hash large verified weights twice'
        return original(path)
    monkeypatch.setattr(stages,'sha',checked)
    assert stages.verify_sft_handoff(actor,{weight.name:checksum})['completed_updates']==1500


def test_old_inference_export_is_not_accepted_as_new_stage1(tmp_path,monkeypatch):
    monkeypatch.setattr(stages,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='complete local Stage1'):
        stages.verify_sft_handoff(tmp_path/'assets/route-sft1500')
    model=tmp_path/'runs/sft/checkpoint_001500/model';model.mkdir(parents=True)
    with pytest.raises(ValueError,match='Run Stage1 SFT first'):
        stages.verify_sft_handoff(model)
