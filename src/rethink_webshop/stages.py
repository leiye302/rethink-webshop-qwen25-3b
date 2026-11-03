"""Bind Stage 2 to a complete local Stage-1 model trained on the selected data."""
import hashlib
import json
from pathlib import Path
from .common import ROOT, load, sha


def default_actor(baseline=False):
    if baseline:
        return 'assets/base'
    return f"{load('sft.json')['output_path']}/checkpoint_{load('sft.json')['total_updates']:06d}/model"


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_sft_handoff(actor,weight_hashes=None):
    """Verify provenance and model files, without loading optimizer states or CUDA.

    validate_actor can supply weight hashes it has just computed, so large model
    shards are not hashed twice. This is not an SFT optimizer-resume operation.
    """
    actor=Path(actor).resolve()
    if actor.name!='model' or not actor.is_relative_to(ROOT.resolve()):
        raise ValueError('Stage2 requires a complete local Stage1 checkpoint/model')
    checkpoint=actor.parent
    try:
        manifest=json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))
        identity=json.loads((checkpoint/'identity.json').read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise ValueError('Run Stage1 SFT first; old released inference-only weights are not this dataset run') from error
    recipe,data,model=load('sft.json'),load('data.json'),load('model.json')
    if (manifest.get('schema')!='webshop-immutable-full-checkpoint-v1'
            or manifest.get('complete') is not True
            or manifest.get('completed_updates')!=recipe['total_updates']
            or manifest.get('world_size')!=recipe['prepared_world_size']):
        raise ValueError('Stage1 must finish all updates with a complete matching-world-size checkpoint')
    config=identity.get('config',{})
    if (any(config.get(k)!=v for k,v in recipe.items())
            or identity.get('config_sha256')!=_digest(config)
            or manifest.get('identity_sha256')!=_digest(identity)):
        raise ValueError('Stage1 configuration/identity differs from the selected recipe')
    if identity.get('data',{}).get('files')!=data['files']:
        raise ValueError('Stage1 data differs: retrain on the selected buy-v9 dataset; never reuse old SFT weights')
    parent=identity.get('parent',{})
    if (parent.get('revision')!=model['revision']
            or parent.get('model_lock_sha256')!=sha(ROOT/'configs/model.lock.json')):
        raise ValueError('Stage1 parent model differs from the selected 3B base')
    records=manifest.get('files',{})
    for name in ('identity.json','config.json'):
        target=checkpoint/name
        record=records.get(name,{})
        if (not target.is_file() or record.get('bytes')!=target.stat().st_size
                or record.get('sha256')!=sha(target)):
            raise ValueError('Stage1 checkpoint metadata failed its SHA/size check')
    if json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('Stage1 saved configuration differs from its identity')
    model_records={name:record for name,record in records.items() if name.startswith('model/')}
    actual={p.relative_to(checkpoint).as_posix() for p in actor.rglob('*') if p.is_file()}
    if not model_records or set(model_records)!=actual:
        raise ValueError('Stage1 model file inventory differs from its complete checkpoint')
    for name,record in model_records.items():
        target=checkpoint/name
        if not target.resolve().is_relative_to(actor) or target.is_symlink():
            raise ValueError('Stage1 model contains linked/escaped files')
        checksum=(weight_hashes or {}).get(target.name) if target.parent==actor else None
        checksum=checksum or sha(target)
        if record.get('bytes')!=target.stat().st_size or record.get('sha256')!=checksum:
            raise ValueError('Stage1 model file failed its SHA/size check')
    return {'completed_updates':manifest['completed_updates'],
        'identity_sha256':manifest['identity_sha256'],'manifest_sha256':sha(checkpoint/'manifest.json'),
        'dataset_revision':data.get('dataset_revision'),
        'train_sha256':data['files']['train.jsonl']['sha256']}
