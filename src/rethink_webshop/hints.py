"""Small SDAR hooks. Route selection is ONLY generated marker token IDs."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from .common import ROOT, atomic_json
from .hint_api import fetch


def enabled(config):
    return bool(config.algorithm.get('local_hints',{}).get('enabled',False))


def capture(records, tokenizer, batch, obs, available, text_actions, next_obs, rewards, dones, active, step):
    vocab=tokenizer.get_vocab()
    marker=vocab.get('<RETHINK>')
    if marker is None or tokenizer.encode('<RETHINK>',add_special_tokens=False)!=[marker]:
        raise ValueError('Method Actor must have one registered RETHINK token')
    responses=batch.batch['responses']
    length=responses.shape[1]
    masks=batch.batch['attention_mask'][:,-length:]
    for i in range(len(records)):
        if not bool(active[i]):
            continue
        ids=responses[i][masks[i].bool()].detach().cpu().tolist()
        text=text_actions[i]
        reasoning=re.search(r'<think>([\s\S]*?)</think>',text)
        records[i].append({'step':step,'visible_prompt':obs['text'][i],
              'reasoning':reasoning[1] if reasoning else '', 'actor_response':text,
              'generated_ids':ids,'triggered':marker in ids,
              'available_actions':available[i],
              'feedback':next_obs['text'][i], 'reward':float(rewards[i]),'done':bool(dones[i])})


def attach(records, batch_list):
    for episode,rows in zip(records,batch_list):
        if not episode:
            continue
        payload=json.dumps({'steps':episode},ensure_ascii=False,separators=(',',':'))
        by_step={r['step']:r for r in episode}
        for step,row in enumerate(rows):
            row['rethink_episode_json']=payload
            row['rethink_step']=step
            row['rethink_triggered']=bool(by_step.get(step,{}).get('triggered',False))


def collect_hints(batch,cache,timeout=60,retries=2,workers=4):
    # Metadata follows each row through SDAR's duplication/balancing permutation.
    groups={}; mapping=[]
    for raw,step,triggered in zip(batch.non_tensor_batch['rethink_episode_json'],
            batch.non_tensor_batch['rethink_step'],batch.non_tensor_batch['rethink_triggered']):
        if triggered:
            groups.setdefault(str(raw),json.loads(str(raw)))
        mapping.append((str(raw),int(step),bool(triggered)))
    def one(item):
        raw,episode=item
        try:
            return raw,fetch(episode,cache,timeout=timeout,retries=retries)
        except Exception:
            # A non-algorithm API/helper failure must not abort policy training.
            return raw,({}, {'episode':'hint_helper_failure'})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=dict(pool.map(one,groups.items()))
    hints=[];reasons=[]
    for raw,step,triggered in mapping:
        if not triggered:
            hints.append(None); reasons.append('not_triggered'); continue
        values,failures=results[raw]
        hints.append(values.get(step))
        reasons.append('applied' if step in values else failures.get(step,failures.get('episode','missing_step')))
    return hints,reasons


class RowSkillProvider:
    """Delegate native skill selection unchanged, append only this row's hint."""
    def __init__(self,parent,hint):
        self.parent=parent; self.hint=hint; self.candidate=None

    def __getattr__(self,name):
        original=getattr(self.parent,name)
        if not name.startswith('get_privileged_info'):
            return original
        def wrapped(*args,**kwargs):
            static=original(*args,**kwargs)
            self.candidate=static+'\n\n[Local reasoning hint for this decision]\n'+self.hint if self.hint else static
            return self.candidate
        return wrapped


def score_teacher(trainer,batch):
    import torch
    from verl import DataProto
    from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
    options=trainer.config.algorithm.local_hints
    cache=ROOT/'runs/hint_cache'
    begin=time.perf_counter()
    hints,reasons=collect_hints(batch,cache,timeout=int(options.get('timeout_seconds',60)),
            retries=int(options.get('retries',2)),workers=int(options.get('workers',4)))
    max_length=trainer.config.data.max_prompt_length
    parts=[]
    for i,hint in enumerate(hints):
        row=batch.select_idxs([i])
        provider=RowSkillProvider(trainer.skill_provider,hint)
        teacher=build_teacher_batch(row,provider,trainer.tokenizer,max_length,
                                   truncation=trainer.config.data.get('truncation','error'))
        if hint:
            responses=row.batch['responses'].shape[1]
            prompt=row.batch['input_ids'][0,:-responses]
            mask=row.batch['attention_mask'][0,:-responses].bool()
            text=trainer.tokenizer.decode(prompt[mask],skip_special_tokens=False)
            candidate='[Privileged Skill Information]\n'+provider.candidate+'\n\n'+text
            if len(trainer.tokenizer.encode(candidate,add_special_tokens=False))>max_length:
                # Never silently truncate an applied local hint/expand Student budget.
                reasons[i]='teacher_prompt_budget'
                teacher=build_teacher_batch(row,trainer.skill_provider,trainer.tokenizer,max_length)
        if not torch.equal(teacher.batch['responses'],row.batch['responses']):
            raise AssertionError('Teacher must score unchanged sampled Student responses')
        parts.append(teacher)
    teacher_batch=DataProto.concat(parts)
    # SDAR's SAME CURRENT Actor, no gradients, not a separately frozen Teacher.
    result=trainer.actor_rollout_wg.compute_log_prob(teacher_batch).batch['old_log_probs']
    report={'step':trainer.global_steps,'decisions':len(hints),
            'actor_marker_positions':sum(x!='not_triggered' for x in reasons),
            'applied':reasons.count('applied'),'fallback_reasons':{x:reasons.count(x) for x in sorted(set(reasons))},
            'seconds':time.perf_counter()-begin,'analyzer_is_router':False,'student_sees_hints':False}
    run_dir=options.get('audit_dir',str(ROOT/'runs/opsd/hints'))
    atomic_json(__import__('pathlib').Path(run_dir)/f'update_{trainer.global_steps:06d}.json',report)
    print('[RETHINK hint coverage] '+json.dumps(report),flush=True)
    return result
