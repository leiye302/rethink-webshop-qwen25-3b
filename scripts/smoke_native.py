"""Exercise the actual shared140 CPU worker, including per-task seeds; no API/model."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/sft')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from rethink_webshop.fixed_validation import manifest
from rethink_webshop.setup import verify_framework
from native_worker import NativeSessionWorker

verify_framework()
fixed=manifest()
selected=[fixed['selection'][i] for i in (0,1,139)]
worker=NativeSessionWorker(ROOT)
try:
    assert worker.ready['identity']['products']==1000
    assert worker.ready['identity']['environment_seed']==1000
    items=[dict(session_id='cpu_smoke_'+str(row['worker']),
                task_id=row['task_id'],worker_seed=row['worker_seed'],
                goal_index=row['goal_index']) for row in selected]
    states=worker.reset(items)['items']
    assert [row['task_id'] for row in states]==[row['task_id'] for row in selected]
    assert all(row['state']['available_actions']['has_search_bar'] for row in states)
    steps=[dict(session_id=row['session_id'],
                response='<think>Search for a candidate.</think><action>search[shirt]</action>',
                expected_state_sha256=row['state']['state_sha256']) for row in states]
    result=worker.step(steps)['items']
    assert len(result)==3
    assert all(row['result']['native_format_valid'] for row in result)
    assert all(row['result']['next_observation'] for row in result)
    assert all(row['next_state'] is not None for row in result if not row['result']['done'])
    print(json.dumps({'passed':True,'cpu_only':True,'api_called':False,
          'model_loaded':False,'task_ids':[r['task_id'] for r in states],
          'seeds':[r['worker_seed'] for r in selected],
          'native_state_feedback_verified':True,
          'environment':worker.ready['identity']},ensure_ascii=False))
finally:
    worker.close()
