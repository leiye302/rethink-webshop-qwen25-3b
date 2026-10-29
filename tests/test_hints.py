import json
import os
from pathlib import Path
import pytest
from rethink_webshop import hint_api


@pytest.fixture
def episode():
    return {'steps':[{'step':0,'triggered':True,'available_actions':{'has_search_bar':False,'clickables':['Buy Now','< Prev']}},
                     {'step':1,'triggered':True,'available_actions':{'has_search_bar':True,'clickables':['Back to Search']}}]}


def test_partial_validation_keeps_other_valid_steps(episode):
    raw={'hints':[{'step':0,'hint':'The current selection meets the visible goal.','next_action':'click[Buy Now]'},
                  {'step':1,'hint':'Invent a button','next_action':'click[not here]'}]}
    valid,why=hint_api.validate_hints(raw,episode)
    assert set(valid)=={0} and why=={1:'illegal_next_action'}


def test_duplicates_are_not_silently_accepted(episode):
    row={'step':0,'hint':'Purchase here.','next_action':'click[Buy Now]'}
    valid,why=hint_api.validate_hints({'hints':[row,row,row]},episode)
    assert not valid and why[0]=='duplicate_step'


@pytest.mark.parametrize('content',[
    '{"hints":[]}', '```json\n{"hints":[]}\n```',
    [{'type':'text','text':'{"hints":[]}'}]])
def test_common_compatible_content_types(content):
    assert hint_api.extract_content({'choices':[{'message':{'content':content}}]})=={'hints':[]}


def test_wrong_types_and_empty_responses_fail_closed():
    for value in [None,{},123,[], '']:
        with pytest.raises((ValueError,TypeError)):
            hint_api.extract_content({'choices':[{'message':{'content':value}}]})


def test_bounded_failure_returns_normal_teacher_fallback(tmp_path,monkeypatch,episode):
    monkeypatch.setattr(hint_api,'auth',lambda:{'base':'https://provider.example/v1','key':'test-placeholder','model':'chosen-model'})
    calls=[]
    def failed(*args,**kwargs):
        calls.append(1)
        raise TimeoutError('must not leak a response body')
    result,why=hint_api.fetch(episode,tmp_path,opener=failed,sleeper=lambda _:None)
    assert result=={} and len(calls)==3
    assert why=={'episode':'transport_or_response_invalid'}


def test_auth_failure_does_not_loop(tmp_path,monkeypatch,episode):
    import urllib.error
    monkeypatch.setattr(hint_api,'auth',lambda:{'base':'https://provider.example/v1','key':'placeholder','model':'chosen-model'})
    calls=[]
    def failed(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://provider.example',401,'bad',{},None)
    assert hint_api.fetch(episode,tmp_path,opener=failed,sleeper=lambda _:None)[0]=={}
    assert len(calls)==1


def test_native_navigation_and_search_grammar():
    available={'has_search_bar':True,'clickables':['< Prev','navy','Buy Now']}
    assert hint_api.allowed('click[< Prev]',available)
    assert hint_api.allowed('search[blue cotton shirt]',available)
    assert not hint_api.allowed('search[<q>hidden]',available)
    assert not hint_api.allowed('click[not visible]',available)
    assert not hint_api.allowed('search[x]',{'has_search_bar':False,'clickables':[]})


def test_hint_coverage_follows_row_identity_after_permutation(monkeypatch,episode,tmp_path):
    from rethink_webshop import hints
    raw=json.dumps(episode)
    class Batch:
        non_tensor_batch={'rethink_episode_json':[raw,raw,raw],
                          'rethink_step':[1,0,1],'rethink_triggered':[True,False,True]}
    calls=[]
    def request(ep,*args,**kwargs):
        calls.append(ep)
        return {1:'local hint for step 1'},{}
    monkeypatch.setattr(hints,'fetch',request)
    text,why=hints.collect_hints(Batch(),tmp_path)
    assert text==['local hint for step 1',None,'local hint for step 1']
    assert len(calls)==1 and why==['applied','not_triggered','applied']
