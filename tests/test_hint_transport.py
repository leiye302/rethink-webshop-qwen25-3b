import json
from pathlib import Path
import pytest
from rethink_webshop import hint_api


def test_auth_rejects_wrong_hint_model(tmp_path,monkeypatch):
    p=tmp_path/'auth.json'
    p.write_text(json.dumps({'base':'https://provider.example/v1','key':'placeholder',
                            'model':'an-annotation-model-not-the-training-recipe'}))
    p.chmod(0o600);monkeypatch.setenv('RETHINK_HINT_AUTH',str(p))
    with pytest.raises(ValueError,match='ALFWorld-final'):
        hint_api.auth()


def test_no_provider_switch_or_billing_endpoint_in_transport():
    source=Path(hint_api.__file__).read_text()
    assert '/zen/v1' not in source and 'mimo' not in source.lower()
    assert hint_api.MODEL=='glm-5.3-flash'


def test_normalizer_uses_raw_preaction_snapshot_and_executed_action():
    episode={'task':'Goal','steps':[{'step':2,'triggered':True,
        'visible_prompt':'Human step 3 / history','observation':'Raw page before action',
        'reasoning':'Actual generated reasoning','actor_response':'not used',
        'action':'click[Buy Now]','feedback':'Only afterwards',
        'available_actions':{'has_search_bar':False,'clickables':['Buy Now']}}]}
    group=hint_api.prepare_group(episode)
    assert group['targets']=={'2':['2']}
    step=group['episode']['steps'][0]
    assert step['observation']=='Raw page before action' and step['action']=='click[Buy Now]'
    assert step['feedback']=='Only afterwards'
    assert group['episode']['benchmark']=='webshop'


def test_protocol_and_body_are_pinned_to_alfworld_final():
    body=hint_api.protocol.request_body({
        'key':'test','targets':{},'episode':{'benchmark':'webshop','task':'Goal','steps':[]}})
    assert body['model']=='glm-5.3-flash'
    assert body['max_tokens']==1024 and body['reasoning_effort']=='low'
    assert body['response_format']=={'type':'json_object'}
