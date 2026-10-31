import json
import pytest
from rethink_webshop.run import complete_weight_files

def test_missing_shard_is_rejected_before_training(tmp_path):
    (tmp_path/'model-00001-of-00002.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError,match='weight index'):
        complete_weight_files(tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
        'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}))
    with pytest.raises(ValueError,match='Incomplete'):
        complete_weight_files(tmp_path)

def test_complete_named_shard_set_passes_file_presence_guard(tmp_path):
    names=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
    for name in names:(tmp_path/name).write_bytes(b'fixture')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':dict(zip(['a','b'],names))}))
    assert [p.name for p in complete_weight_files(tmp_path)]==names
