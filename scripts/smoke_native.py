"""A real CPU-only native reset/action/feedback check; no model or API."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''
from webshop_data.native import NativeWorld
from rethink_webshop.common import UPSTREAM,WEBSHOP,load
from rethink_webshop.setup import verify_framework

verify_framework()
world=NativeWorld(UPSTREAM,WEBSHOP/'data')
env=None
try:
    assert world.identity['products']==1000 and world.identity['environment_seed']==0
    goal=load('eval128.json')['task_ids'][0]
    env,view,obs=world.reset(goal,'cpu_public_smoke')
    state=world.visible(env,view,obs)
    assert state['available_actions']['has_search_bar']
    step=world.step(env,view,state['formatted_observation'],
         '<think>I will search for a candidate product.</think><action>search[shirt]</action>')
    assert step['native_format_valid'] and step['next_observation']
    if not step['done']:
        next_state=world.visible(env,view,step['next_observation'])
        assert next_state['prompt']!=state['prompt']
    print(json.dumps({'passed':True,'cpu_only':True,'real_native_environment':True,
           'model_loaded':False,'api_called':False,'task_id':goal,'identity':world.identity,
           'executed_action':step['environment_action'],'feedback_returned':True},ensure_ascii=False))
finally:
    if env is not None: env.close()
    world.base.close()
