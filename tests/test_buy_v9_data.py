"""The active 3B data is the same published buy-v9 profile available to 1.5B."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]
VERSION=ROOT/'data/buy_v9'


def test_active_v9_bytes_counts_and_natural_good_actions():
    spec=json.loads((ROOT/'configs/data.json').read_text())
    assert spec==json.loads((VERSION/'data.json').read_text())
    assert spec['dataset_revision']=='buy-v9'
    bundle=ROOT/'data/frozen_sft.zip'
    assert bundle.read_bytes()==(VERSION/'frozen_sft.zip').read_bytes()
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==spec['bundle_sha256']
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist())==set(spec['files'])|{'manifest.json'}
        assert json.loads(archive.read('manifest.json'))['files']==spec['files']
        for name,record in spec['files'].items():
            data=archive.read(name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        rows=[json.loads(x) for x in archive.read('train.jsonl').splitlines() if x]
    assert Counter(x['route'] for x in rows)=={'good':7500,'bad':4500}
    assert Counter(x['route'] for x in rows if x['supervision_action'].lower()=='click[buy now]')=={'good':176,'bad':92}
    assert all(x['supervision_action']==x['source_executed_action'] for x in rows if x['route']=='good')
    assert spec['files']['val.jsonl']['sha256']=='7d344341478091fd5bb9e592bcd25f4e563d12ac05131b5234fd79ad30d85fd1'
    assert spec['files']['generation_val200.jsonl']['sha256']=='85d40da59056ee405d4b99f5cad044f46d0567a3b405efcd97cd0ad2a18b9047'


def test_new_data_uses_unchanged_actual_training_validator(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from rethink_webshop import setup,common
    import train
    from route_bce import core
    (tmp_path/'configs').mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'configs/data.json',tmp_path/'configs/data.json')
    shutil.copyfile(ROOT/'data/frozen_sft.zip',tmp_path/'data/frozen_sft.zip')
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(setup,'ROOT',tmp_path)
    setup.data()
    cfg=train.load_config(ROOT/'configs/sft.json')
    rows,val,order,identity=train.validate_dataset(tmp_path,cfg,core)
    assert (len(rows),len(val),len(order),identity['generation_rows'])==(12000,902,1500,200)
    assert sorted(i for batch in order for i in batch)==list(range(12000))
    assert all(sum(rows[i]['route']=='good' for i in batch)==5 for batch in order)
    setup.data()
    # A self-consistent extracted dataset must still match the actively selected profile.
    active=tmp_path/'configs/data.json'
    selected=json.loads(active.read_text());selected['files']['train.jsonl']['sha256']='0'*64
    active.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match='selected database'):
        train.validate_dataset(tmp_path,cfg,core)
    shutil.copyfile(ROOT/'configs/data.json',active)
    target=tmp_path/'datasets/frozen_sft/train.jsonl'
    target.write_bytes(target.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Refuse to replace'):setup.data()
