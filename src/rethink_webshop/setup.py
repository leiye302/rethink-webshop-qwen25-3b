"""Prepare pinned public dependencies, frozen data and verified model assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import uuid
import zipfile
from .common import ROOT, UPSTREAM, WEBSHOP, load, sha, atomic_json


def git(path, *args):
    return subprocess.run(['git','-c','core.autocrlf=false','-C',str(path),*args],
                          capture_output=True, text=True, check=True).stdout.strip()


def verify_framework():
    lock=load('upstream.json')
    if git(UPSTREAM,'rev-parse','HEAD') != lock['commit']:
        raise ValueError('Wrong SDAR revision')
    if sha(ROOT/'patches/sdar.patch') != lock['patch_sha256']:
        raise ValueError('Unexpected patch')
    changed=set(git(UPSTREAM,'diff','--name-only','HEAD').splitlines())
    if changed != set(lock['patched_files']):
        raise ValueError('Unexpected modified upstream files')
    git(UPSTREAM,'apply','--reverse','--check',str(ROOT/'patches/sdar.patch'))
    for name, expected in {**lock['unchanged_files'], **lock['patched_files']}.items():
        if sha(UPSTREAM/name) != expected:
            raise ValueError('SDAR byte hash differs: '+name)
    return lock


def framework(source=None):
    if UPSTREAM.exists():
        verify_framework(); return
    lock=load('upstream.json')
    pending=UPSTREAM.with_name('sdar-'+uuid.uuid4().hex[:10])
    pending.parent.mkdir(parents=True,exist_ok=True)
    if source:
        git(ROOT,'clone','--no-checkout','--no-hardlinks',str(Path(source).resolve()),str(pending))
        git(pending,'checkout','--detach',lock['commit'])
        git(pending,'remote','set-url','origin',lock['repository'])
    else:
        pending.mkdir()
        git(pending,'init')
        git(pending,'remote','add','origin',lock['repository'])
        git(pending,'fetch','--depth','1','origin',lock['commit'])
        git(pending,'checkout','--detach','FETCH_HEAD')
    patch=ROOT/'patches/sdar.patch'
    if sha(patch)!=lock['patch_sha256']:
        raise ValueError('Patch hash mismatch')
    git(pending,'apply','--check',str(patch)); git(pending,'apply',str(patch))
    pending.rename(UPSTREAM)
    verify_framework()


def data():
    lock=load('data.json')
    bundle=ROOT/'data/frozen_sft.zip'
    if sha(bundle)!=lock['bundle_sha256']:
        raise ValueError('Data archive changed')
    target=ROOT/'datasets/frozen_sft'
    target.mkdir(parents=True,exist_ok=True)
    allowed=set(lock['files'])|{'manifest.json'}
    with zipfile.ZipFile(bundle) as z:
        if set(z.namelist())!=allowed or len(z.namelist())!=len(allowed):
            raise ValueError('Unexpected ZIP members')
        for n in sorted(allowed):
            content=z.read(n)
            if n in lock['files']:
                check_bytes(content,lock['files'][n])
            out=target/n
            if out.exists() and out.read_bytes()!=content:
                raise ValueError('Refuse to replace modified accepted data')
            out.write_bytes(content)


def check_bytes(content, spec):
    if len(content)!=spec['bytes']:
        raise ValueError('Partial resource')
    if spec.get('sha256') and hashlib.sha256(content).hexdigest()!=spec['sha256']:
        raise ValueError('Resource SHA256 differs')
    if spec.get('git_blob'):
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=spec['git_blob']:
            raise ValueError('Resource Git blob differs')


def base(accept_license=False, tokenizer_only=False):
    from huggingface_hub import hf_hub_download
    lock=load('model.lock.json')
    if lock['license']=='qwen-research' and not accept_license:
        raise PermissionError('Read Qwen Research License; pass --accept-model-license to accept it')
    target=ROOT/'assets/base'; target.mkdir(parents=True,exist_ok=True)
    receipt={}
    for name,spec in lock['files'].items():
        if tokenizer_only and ('safetensors' in name or name=='config.json'):
            continue
        path=Path(hf_hub_download(lock['model_id'],name,revision=lock['revision'],local_dir=target))
        if spec.get('sha256'):
            if path.stat().st_size!=spec['bytes'] or sha(path)!=spec['sha256']:
                raise ValueError('Model file SHA differs: '+name)
        else:
            check_bytes(path.read_bytes(),spec)
        receipt[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
    if not tokenizer_only:
        atomic_json(target/'verified_model.json',{'revision':lock['revision'],'files':receipt})


def checkpoint(accept_license=False):
    if not accept_license:
        raise PermissionError('The released 3B weights retain Qwen Research License; explicitly accept it')
    from .assets import download_file
    lock=load('checkpoint1500.json')
    target=ROOT/'assets/route-sft1500'; target.mkdir(parents=True,exist_ok=True)
    prefix=f'https://github.com/{lock["repository"]}/releases/download/{lock["tag"]}'
    for n,spec in lock['assets'].items():
        download_file(target/n,[prefix+'/'+n],spec)
    manifest=json.loads((target/'checkpoint_manifest.json').read_text())
    if manifest['sft_updates']!=1500:
        raise ValueError('Wrong published checkpoint')
    atomic_json(target/'download_verified.json',{'step':1500,'release':lock['tag'],'files':lock['assets']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework',action='store_true')
    p.add_argument('--source',type=Path,help='Optional local public Git mirror, never required')
    p.add_argument('--data',action='store_true')
    p.add_argument('--base',action='store_true')
    p.add_argument('--tokenizer-only',action='store_true')
    p.add_argument('--checkpoint1500',action='store_true')
    p.add_argument('--accept-model-license',action='store_true')
    p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.framework: framework(a.source)
    if a.data: data()
    if a.base or a.tokenizer_only: base(a.accept_model_license,a.tokenizer_only)
    if a.checkpoint1500: checkpoint(a.accept_model_license)
    if a.verify: verify_framework()
    print('Preparation complete. No training or paid API request was started.')


if __name__=='__main__':
    main()
