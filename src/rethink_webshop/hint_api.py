"""ALFWorld-final GLM hint protocol, with only native shopping state/actions adapted."""
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from .common import atomic_json
from . import rethink_api_protocol as protocol
from .rethink_benchmark_contract import command_allowed

MODEL=protocol.MODEL
_SESSION_ID='webshop-glm-hints-'+uuid.uuid4().hex


class HintAPIHold(RuntimeError):
    """Retain the same rollout/pre-update Actor; never silently discard hints."""
    def __init__(self,reason,temporary=False):
        self.reason=reason
        self.temporary=temporary
        super().__init__(reason)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise HintAPIHold('api_redirect_refused')


def commands(available):
    if isinstance(available,list):return list(available)
    result=['search[<your query>]'] if available.get('has_search_bar') else []
    result.extend('click['+str(name)+']' for name in available.get('clickables',[]))
    return result


def allowed(action,available):
    return command_allowed(action,commands(available),'webshop')


def prepare_group(episode):
    steps=[]
    for step in episode['steps']:
        action=step.get('action')
        if action is None:
            match=re.search(r'<action>([^\r\n]+?)</action>',step.get('actor_response',''),re.I)
            action=match[1].strip().lower() if match else ''
        steps.append({'step':int(step['step']),
                      'observation':step.get('observation',step.get('visible_prompt','')),
                      'admissible_actions':step.get('admissible_actions',commands(step.get('available_actions',{}))),
                      'reasoning':step.get('reasoning',''),'action':action,
                      'feedback':step.get('feedback','')})
    canonical={'benchmark':'webshop','task':episode.get('task',episode['steps'][0].get('visible_prompt','') if steps else ''),
               'steps':steps}
    targets={str(int(s['step'])):[str(int(s['step']))] for s in episode['steps'] if s.get('triggered')}
    if len({s['step'] for s in steps})!=len(steps):
        raise ValueError('Duplicate captured step ID')
    return {'key':protocol.digest(canonical),'episode':canonical,'targets':targets}


def extract_content(data):
    choice=data['choices'][0]
    content=choice['message'].get('content')
    if isinstance(content,list):
        content=''.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type') in ('text','output_text'))
    if not isinstance(content,str) or not content.strip():raise ValueError('empty_content')
    return content


def parsed_hints(content,group,finish_reason):
    outputs=protocol.parse_hints(content,group,finish_reason)
    valid={};failures={}
    for key,record in outputs.items():
        if record['payload'] is not None:
            valid[int(key)]='Reasoning skill: '+record['payload']['reasoning_skill'].strip()
        else:
            failures[int(key)]=record['error']
    return valid,failures


def validate_hints(raw,episode):
    return parsed_hints(json.dumps(raw,ensure_ascii=False),prepare_group(episode),'stop')


def auth():
    path=Path(os.environ['RETHINK_HINT_AUTH'])
    if os.name!='nt' and path.stat().st_mode & 0o077:
        raise PermissionError('Hint credential file must be owner-only')
    cfg=json.loads(path.read_text())
    parts=urllib.parse.urlsplit(cfg['base'])
    if parts.scheme!='https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Use a credential-free HTTPS API base URL')
    if not cfg.get('key') or cfg.get('model')!=MODEL:
        raise ValueError('ALFWorld-final recipe requires the explicit glm-5.3-flash model')
    return cfg


def fetch(episode,cache_dir,timeout=120,retries=5,opener=None,sleeper=time.sleep):
    """Same final model/body/parser and six-attempt transport boundary as ALFWorld.

    Invalid returned hints use the ordinary static-skill Teacher. Transport,
    auth and quota failures HOLD this batch instead of dropping local hints.
    """
    try:cfg=auth()
    except (KeyError,OSError,ValueError,PermissionError):
        raise HintAPIHold('api_not_configured') from None
    group=prepare_group(episode)
    if not group['targets']:return {},{}
    body=protocol.request_body(group)
    url=cfg['base'].rstrip('/')+'/chat/completions'
    identity=protocol.digest({'body':body,'protocol':protocol.VERSION,'endpoint':url})
    cache=Path(cache_dir)/(identity+'.json')
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['request_sha256']!=identity:raise RuntimeError('Hint cache identity mismatch')
        return parsed_hints(saved.get('content'),group,saved.get('finish_reason'))
    headers={'Content-Type':'application/json','Accept':'application/json',
             'User-Agent':'AlfworldHintEvaluation/1.0','Authorization':'Bearer '+cfg['key']}
    if urllib.parse.urlsplit(cfg['base']).hostname=='opencode.ai' and '/inference/go/' in url:
        headers['x-opencode-session']=_SESSION_ID
    opener=opener or urllib.request.build_opener(NoRedirect()).open
    reason='api_transport_error'
    for attempt in range(retries+1):
        try:
            req=urllib.request.Request(url,json.dumps(body,ensure_ascii=False).encode(),headers)
            with opener(req,timeout=timeout) as response:
                raw=response.read(4*1024**2+1)
            if len(raw)>4*1024**2:raise HintAPIHold('api_response_too_large')
            data=json.loads(raw)
            if not isinstance(data,dict) or not isinstance(data.get('choices'),list) or not data['choices']:
                raise HintAPIHold('api_malformed_response')
            try:
                content=extract_content(data)
                finish=data['choices'][0].get('finish_reason')
            except (KeyError,IndexError,TypeError,ValueError):
                content=None
                finish=data['choices'][0].get('finish_reason') if isinstance(data['choices'][0],dict) else None
            result=parsed_hints(content,group,finish)
            atomic_json(cache,{'request_sha256':identity,'schema':protocol.VERSION,'model':MODEL,
                              'content':content,'finish_reason':finish,'usage':data.get('usage',{})})
            return result
        except urllib.error.HTTPError as exc:
            reason='http_'+str(exc.code)
            if exc.code not in (408,429,500,502,503,504):
                raise HintAPIHold(reason) from None
        except (OSError,urllib.error.URLError,ConnectionError,http.client.RemoteDisconnected,
                http.client.IncompleteRead,json.JSONDecodeError):
            reason='api_transport_error'
        if attempt<retries:sleeper(min(60,2**(attempt+1)))
    raise HintAPIHold(reason,temporary=True)
