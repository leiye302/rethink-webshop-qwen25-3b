"""Explicit native SDAR launch; no secrets in command arguments or Hydra config."""
import argparse
import json
import os
from pathlib import Path
import sys
from .common import ROOT, UPSTREAM, confined, load, atomic_json, sha
from .hardware import WORLD_SIZE, check_cuda, validate_visible_devices


def placeholders():
    """Same text-only placeholder rows as SDAR prepare.py, no unrelated images."""
    import pandas as pd
    target=ROOT/'datasets/rl/text'; target.mkdir(parents=True,exist_ok=True)
    for split,count,name in [('train',16,'train.parquet'),('test',128,'test.parquet')]:
        rows=[{'data_source':'text','prompt':[{'role':'user','content':''}],
               'ability':'agent','extra_info':{'split':split,'index':i}} for i in range(count)]
        pd.DataFrame(rows).to_parquet(target/name,index=False)


def resolved(actor,output,baseline=False):
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(UPSTREAM/'verl/trainer/config/ppo_trainer.yaml')
    settings=load('opsd.json')
    # Only named settings; model/paths are resolved for this clean checkout.
    overrides=dict(settings)
    overrides.update({'data.train_files':str(ROOT/'datasets/rl/text/train.parquet'),
       'data.val_files':str(ROOT/'datasets/rl/text/test.parquet'),
       'actor_rollout_ref.model.path':str(actor),
       'algorithm.sdar.skills_dir':str(UPSTREAM/'skills/webshop'),
       'algorithm.local_hints.enabled':not baseline,
       'algorithm.local_hints.audit_dir':str(output/'hints'),
       'trainer.default_local_dir':str(output/'checkpoints'),
       'trainer.rollout_data_dir':str(output/'rollouts'),
       'trainer.validation_data_dir':str(output/'validation'),
       'trainer.experiment_name':'baseline' if baseline else 'rethink',
       'ray_init.address':'local',
       'ray_init.runtime_env.env_vars.PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(UPSTREAM)]),
       'ray_init.runtime_env.env_vars.JAVA_HOME':str(ROOT/'assets/java'),
       'ray_init.runtime_env.env_vars.JVM_PATH':str(ROOT/'assets/java/lib/server/libjvm.so')})
    for key,value in overrides.items():
        OmegaConf.update(cfg,key,value,merge=False,force_add=True)
    return cfg


def validate_actor(path,baseline):
    from transformers import AutoConfig,AutoTokenizer
    profile=load('model.json')
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    expected=1536 if profile['size']=='1.5b' else 2048
    if config.model_type!='qwen2' or config.hidden_size!=expected:
        raise ValueError('Actor is not the selected Qwen2.5 model size')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    if not baseline:
        token=tok.get_vocab().get('<RETHINK>')
        if token is None or tok.encode('<RETHINK>',add_special_tokens=False)!=[token] or token>=config.vocab_size:
            raise ValueError('Registered Actor marker is missing or out of model vocabulary')
    weights=sorted(Path(path).glob('*.safetensors'))
    if not weights:
        raise ValueError('Complete local safetensors weights are required')
    files={p.name:sha(p) for p in weights}
    return {'model_config_sha256':sha(Path(path)/'config.json'),'weight_sha256':files,
            'tokenizer_sha256':sha(Path(path)/'tokenizer.json'),
            'baseline':baseline,'model_size':profile['size']}


def resume_checkpoint(output):
    directory=output/'checkpoints'
    tracker=directory/'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        raise ValueError('No native resume tracker; refusing to silently restart from base')
    text=tracker.read_text().strip()
    if not text.isdigit() or not 0<int(text)<=150:
        raise ValueError('Invalid native resume step')
    checkpoint=directory/f'global_step_{int(text)}'
    required=[checkpoint/'data.pt']
    required += [checkpoint/'actor'/f'{kind}_world_size_{WORLD_SIZE}_rank_{rank}.pt'
                 for rank in range(WORLD_SIZE) for kind in ['model','optim','extra_state']]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError('Missing native model/optimizer/RNG/dataloader resume files')
    return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',action='store_true',help='Same public GRPO+OPSD beta5 from base, no local hints')
    p.add_argument('--actor',help='Project-relative Actor override; default is released/locally trained SFT1500')
    p.add_argument('--run-dir',default='runs/opsd')
    p.add_argument('--allow-api-cost',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    from .setup import verify_framework
    verify_framework(); placeholders()
    size=load('model.json')['size']
    default='assets/base' if a.baseline else ('assets/route-sft1500' if size=='3b' else 'runs/sft/checkpoint_001500/model')
    actor=confined(a.actor or default)
    output=confined(a.run_dir)
    cfg=resolved(actor,output,a.baseline)
    from omegaconf import OmegaConf
    if a.prepare_only:
        output.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(cfg,output/'prepared_config.yaml')
        print('Native SDAR configuration prepared; no model, API or CUDA was used.'); return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Explicitly set the GPUs allocated to this job')
    validate_visible_devices(os.environ['CUDA_VISIBLE_DEVICES'])
    import torch
    hardware=check_cuda(torch)
    from .assets import verify as verify_assets
    verify_assets()
    identity=validate_actor(actor,a.baseline)
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/'launch_identity.json'
    if receipt.exists() and not a.resume:
        raise FileExistsError('This run already exists; explicitly --resume it or use a new run directory')
    identity.update({'recipe_sha256':sha(ROOT/'configs/opsd.json'),'upstream_sha256':sha(ROOT/'configs/upstream.json'),
                     'actor':str(actor.relative_to(ROOT)),'validation_sha256':sha(ROOT/'configs/eval128.json'),
                     'world_size':WORLD_SIZE,'rollout_tensor_parallel':2})
    if a.resume and (not receipt.exists() or json.loads(receipt.read_text())!=identity):
        raise ValueError('Resume Actor/recipe/validation identity mismatch')
    private=None
    if not a.baseline:
        if not a.allow_api_cost:
            p.error('Method uses an external API: --allow-api-cost is required')
        if not all(os.environ.get(k) for k in ['HINT_API_BASE','HINT_MODEL','HINT_API_KEY']):
            p.error('Set HINT_API_BASE, HINT_MODEL and HINT_API_KEY in environment, not CLI/config')
        private=output/'private/hint_auth.json'; private.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        fd=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump({'base':os.environ['HINT_API_BASE'],'model':os.environ['HINT_MODEL'],'key':os.environ.pop('HINT_API_KEY')},f)
        os.chmod(private,0o600)
        os.environ['RETHINK_HINT_AUTH']=str(private)
        from .hint_api import auth
        auth()  # Validate without a paid request.
        OmegaConf.update(cfg,'ray_init.runtime_env.env_vars.RETHINK_HINT_AUTH',str(private),force_add=True)
    cfg.trainer.resume_mode='resume_path' if a.resume else 'disable'
    if a.resume:
        cfg.trainer.resume_from_path=str(resume_checkpoint(output))
    os.environ.update({'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(UPSTREAM)]),
             'JAVA_HOME':str(ROOT/'assets/java'),'JVM_PATH':str(ROOT/'assets/java/lib/server/libjvm.so'),
             'TOKENIZERS_PARALLELISM':'false','RAY_ADDRESS':'local','VLLM_USE_V1':'1'})
    atomic_json(receipt,identity)
    atomic_json(output/'hardware.json',{'profile':'4xA100-80GB','devices':hardware,
                                      'rollout_tp':2,'rollout_dp':2,'fsdp_world_size':WORLD_SIZE})
    OmegaConf.save(cfg,output/'resolved_config.yaml')
    sys.path.insert(0,str(UPSTREAM)); os.chdir(UPSTREAM)
    from verl.trainer.main_sdar import run_sdar
    import ray
    try:
        run_sdar(cfg)
    finally:
        ray.shutdown()
        if private and private.is_file():
            private.unlink()


if __name__=='__main__':
    main()
