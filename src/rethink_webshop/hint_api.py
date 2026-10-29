"""Bounded, cached text API. Never judges routes or supplies Student actions."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from .common import atomic_json

SYSTEM = '''You provide TRAINING-ONLY local reasoning hints for a shopping agent.
The Actor has ALREADY chosen the requested steps by emitting <RETHINK>.
Do not judge Good/Bad, reject a route, invent observations, or redo the task.
You receive a complete rollout retrospectively. For each requested step, use
its pre-action visible state to explain one local factual/planning correction.
Later feedback can diagnose the mistake but do not confuse a later page with
the target step's current page. Recommend one legal NEXT action, not a rollout.
Use only click targets listed at that step; search queries may be rewritten
only if that step has a search bar. Purchase when the visible goal and selected
options justify it; do not invent hidden attributes or a remaining-step budget.
Return JSON only: {"hints":[{"step":integer,"hint":"concise guidance",
"next_action":"search[...] or click[...]"}]}. Include every requested step once.
No quality label. Keep each hint below 120 words. Never change the actual rollout.'''


def extract_content(data):
    message=data['choices'][0]['message']
    content=message.get('content')
    if isinstance(content,list):
        content=''.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type') in ('text','output_text'))
    if not isinstance(content,str) or not content.strip():
        raise ValueError('empty_content')
    text=content.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I)
        text=re.sub(r'\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict) or not isinstance(value.get('hints'),list):
        raise ValueError('schema')
    return value


def allowed(action, available):
    if not isinstance(action,str) or '\n' in action or '\r' in action:
        return False
    m=re.fullmatch(r'(click|search)\[([^\r\n]+)\]',action.strip(),re.I)
    if not m or not m[2].strip() or re.search(r'<[aqcr]>',m[2]):
        return False
    if m[1].lower()=='search':
        return bool(available.get('has_search_bar'))
    return m[2].strip().lower() in {str(x).lower() for x in available.get('clickables',[])}


def validate_hints(raw, episode):
    targets={x['step']:x for x in episode['steps'] if x['triggered']}
    valid={}; failures={}; seen=set()
    for obj in raw.get('hints',[]):
        if not isinstance(obj,dict):
            continue
        i=obj.get('step')
        if isinstance(i,bool) or not isinstance(i,int) or i not in targets:
            continue
        if i in seen:
            valid.pop(i,None); failures[i]='duplicate_step'; continue
        seen.add(i)
        hint=obj.get('hint'); command=obj.get('next_action')
        if not isinstance(hint,str) or not hint.strip() or len(hint)>1600:
            failures[i]='hint_text'; continue
        if any(x in hint for x in ('<|im_start|>','<|im_end|>','<RETHINK>')):
            failures[i]='control_token_in_hint'; continue
        if not allowed(command,targets[i]['available_actions']):
            failures[i]='illegal_next_action'; continue
        valid[i]=hint.strip()+'\nSuggested next action: '+command.strip()
    for i in targets:
        if i not in valid:
            failures.setdefault(i,'missing_step')
    return valid,failures


def auth():
    path=Path(os.environ['RETHINK_HINT_AUTH'])
    if os.name!='nt' and path.stat().st_mode & 0o077:
        raise PermissionError('Hint credential file must be owner-only')
    cfg=json.loads(path.read_text())
    parts=urllib.parse.urlsplit(cfg['base'])
    if parts.scheme!='https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Use a credential-free HTTPS API base URL')
    if not cfg.get('key') or not cfg.get('model'):
        raise ValueError('Missing explicitly selected model/key')
    return cfg


def fetch(episode, cache_dir, timeout=60, retries=2, opener=None, sleeper=time.sleep):
    """At most three HTTP/1.1 attempts; every failure has an ordinary-Teacher fallback.

    Cache accepted JSON only, never credentials/raw HTTP bodies. Errors exposed to
    logs are controlled reason codes, not response bodies or exception strings.
    """
    try:
        cfg=auth()
    except (KeyError,OSError,ValueError,PermissionError):
        return {},{'episode':'api_not_configured'}
    body={'model':cfg['model'],'messages':[{'role':'system','content':SYSTEM},
          {'role':'user','content':json.dumps(episode,ensure_ascii=False,separators=(',',':'))}],
          'temperature':0,'max_tokens':min(4096,max(512,220*sum(x['triggered'] for x in episode['steps'])))}
    request_id=hashlib.sha256(json.dumps({'base':cfg['base'],'request':body},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=Path(cache_dir)/(request_id+'.json')
    if cache.exists():
        try:
            return validate_hints(json.loads(cache.read_text()),episode)
        except (OSError,ValueError,TypeError):
            pass
    url=cfg['base'].rstrip('/')+'/chat/completions'
    opener=opener or urllib.request.urlopen
    reason='request_failed'
    for attempt in range(retries+1):
        try:
            req=urllib.request.Request(url,json.dumps(body).encode(),
                {'Content-Type':'application/json','Authorization':'Bearer '+cfg['key']})
            with opener(req,timeout=timeout) as response:
                raw=response.read(4*1024**2+1)
            if len(raw)>4*1024**2:
                raise ValueError('response_too_large')
            result=extract_content(json.loads(raw))
            validated=validate_hints(result,episode)
            atomic_json(cache,result)
            return validated
        except urllib.error.HTTPError as e:
            reason='http_'+str(e.code)
            if e.code in (400,401,402,403,404,429):
                break
        except (OSError,ValueError,KeyError,IndexError,TypeError):
            reason='transport_or_response_invalid'
        if attempt<retries:
            sleeper(min(4,2**attempt))
    return {},{'episode':reason}
