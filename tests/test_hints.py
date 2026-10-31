import io
import json
import urllib.error
import pytest
from rethink_webshop import hint_api


@pytest.fixture
def episode():
    return {'task':'Choose the requested item','steps':[
        {'step':0,'triggered':True,'observation':'Before buy','reasoning':'R0',
         'available_actions':{'has_search_bar':False,'clickables':['Buy Now','< Prev']}},
        {'step':1,'triggered':True,'observation':'Before search','reasoning':'R1',
         'available_actions':{'has_search_bar':True,'clickables':['Back to Search']}}]}


def configure(monkeypatch):
    monkeypatch.setattr(hint_api,'auth',lambda:{
        'base':'https://opencode.ai/inference/go/openai/v1',
        'key':'test-placeholder','model':'glm-5.3-flash'})


def response(content,finish='stop'):
    return io.BytesIO(json.dumps({'choices':[{'finish_reason':finish,
         'message':{'content':json.dumps(content) if isinstance(content,dict) else content}}]}).encode())


def test_final_alfworld_body_and_provider_headers(tmp_path,monkeypatch,episode):
    configure(monkeypatch);calls=[]
    def opener(req,**kwargs):
        body=json.loads(req.data);calls.append(body)
        headers={k.lower():v for k,v in req.header_items()}
        assert headers['user-agent']=='AlfworldHintEvaluation/1.0'
        assert headers['x-opencode-session'].startswith('webshop-glm-hints-')
        assert body['model']=='glm-5.3-flash' and body['temperature']==0
        assert body['reasoning_effort']=='low'
        assert body['max_tokens']==1024+192*2
        assert body['response_format']=={'type':'json_object'}
        assert 'thinking' not in body
        payload=json.loads(body['messages'][1]['content'])
        assert set(payload['output_template']['hints'])=={'0','1'}
        assert [s['step'] for s in payload['requested_targets']]==[0,1]
        assert payload['requested_targets'][0]['observation_before']=='Before buy'
        return response({'hints':{'0':"Next command: 'click[Buy Now]'. Purchase.",
                                  '1':"Next command: 'search[blue cotton shirt]'. Refine."}})
    result,why=hint_api.fetch(episode,tmp_path,opener=opener)
    assert set(result)=={0,1} and why=={} and len(calls)==1
    assert result[0].startswith('Reasoning skill: Next command:')
    assert hint_api.fetch(episode,tmp_path,opener=opener)==(result,why)
    assert len(calls)==1


def test_partial_command_validation_only_rejects_illegal_hint(episode):
    good={0:"Next command: 'click[Buy Now]'. Purchase.",
          1:"Next command: 'click[not here]'. Incorrect."}
    valid,why=hint_api.validate_hints({'hints':{str(k):v for k,v in good.items()}},episode)
    assert set(valid)=={0} and why=={1:'invalid_hint_command_or_length'}


def test_id_coverage_and_schema_match_final_alfworld(episode):
    for raw in [{'hints':[]},{'hints':{'0':"Next command: 'click[Buy Now]'. Purchase."}},
                {'hints':{'0':'x','1':'y','2':'z'}}]:
        valid,why=hint_api.validate_hints(raw,episode)
        assert not valid and set(why)=={0,1}
        assert set(why.values())=={'batch_schema_or_target_coverage'}


def test_duplicate_json_keys_fail_closed(episode):
    group=hint_api.prepare_group(episode)
    text='{"hints":{"0":"x","0":"y","1":"z"}}'
    assert not hint_api.parsed_hints(text,group,'stop')[0]


def test_truncated_json_is_hint_fallback_not_transport_success(tmp_path,monkeypatch,episode):
    configure(monkeypatch)
    result,why=hint_api.fetch(episode,tmp_path,
        opener=lambda *a,**kw:response('{"hints":{','length'))
    assert not result and set(why)=={0,1}
    assert set(why.values())=={'batch_schema_or_target_coverage'}


def test_transport_failure_holds_same_batch_not_silent_hint_drop(tmp_path,monkeypatch,episode):
    configure(monkeypatch);calls=[];delays=[]
    def failed(*a,**kw):
        calls.append(1);raise TimeoutError('Do not leak provider details')
    with pytest.raises(hint_api.HintAPIHold,match='api_transport_error') as caught:
        hint_api.fetch(episode,tmp_path,opener=failed,sleeper=delays.append)
    assert caught.value.temporary and len(calls)==6 and delays==[2,4,8,16,32]


def test_auth_failure_does_not_repeat_request(tmp_path,monkeypatch,episode):
    configure(monkeypatch);calls=[]
    def failed(*a,**kw):
        calls.append(1)
        raise urllib.error.HTTPError('https://provider.example',403,'private body',{},None)
    with pytest.raises(hint_api.HintAPIHold,match='http_403') as caught:
        hint_api.fetch(episode,tmp_path,opener=failed,sleeper=lambda _:None)
    assert not caught.value.temporary and len(calls)==1


def test_search_and_click_use_native_admissibility():
    available={'has_search_bar':True,'clickables':['< Prev','navy','Buy Now']}
    assert hint_api.allowed('click[< Prev]',available)
    assert hint_api.allowed('search[blue cotton shirt]',available)
    assert not hint_api.allowed('search[<search>hidden</search>]',available)
    assert not hint_api.allowed('click[not visible]',available)
    assert not hint_api.allowed('search[x]',{'has_search_bar':False,'clickables':[]})


def test_no_marker_means_no_request(tmp_path,monkeypatch,episode):
    configure(monkeypatch)
    for s in episode['steps']:s['triggered']=False
    def forbidden(*a,**kw):raise AssertionError('Nontriggered episode must not call API')
    assert hint_api.fetch(episode,tmp_path,opener=forbidden)==({},{})


def test_hint_coverage_follows_row_identity_after_permutation(monkeypatch,episode,tmp_path):
    from rethink_webshop import hints
    raw=json.dumps(episode)
    class Batch:
        non_tensor_batch={'rethink_episode_json':[raw,raw,raw],
                          'rethink_step':[1,0,1],'rethink_triggered':[True,False,True]}
    calls=[]
    def request(ep,*args,**kwargs):
        calls.append(ep);return {1:'local hint for step 1'},{}
    monkeypatch.setattr(hints,'fetch',request)
    text,why=hints.collect_hints(Batch(),tmp_path)
    assert text==['local hint for step 1',None,'local hint for step 1']
    assert len(calls)==1 and why==['applied','not_triggered','applied']
