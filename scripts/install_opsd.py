"""Install the pinned runtime without FastAPI's unused command-line extra.

vLLM requires fastapi[standard]. Its CLI extra requires typer>=0.12, whereas
the native WebShop spaCy3.7.2 requires typer<0.10. The vLLM Python API/server
does not use fastapi-cli. Keep every native scoring package fixed; omit ONLY
that unused CLI extra, install all other runtime requirements, then pip check.
No installed metadata or library source is patched.
"""
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]


def requirements_without_cli(requirements,extra=''):
    result=[]
    for text in requirements:
        r=Requirement(text)
        if r.marker and not r.marker.evaluate({'extra':extra}):continue
        if r.name.lower().replace('_','-') in {'fastapi-cli','fastapi-cloud-cli'}:continue
        if r.name.lower()=='fastapi':
            r.extras=set(r.extras)-{'standard'}
        r.marker=None
        result.append(str(r))
    return result


def pip(*args):
    subprocess.run([sys.executable,'-m','pip',*args],check=True)


def main():
    # Metadata can be read without importing vLLM/CUDA.
    pip('install','--no-deps','vllm==0.11.0')
    top=[x.strip() for x in (ROOT/'configs/requirements-opsd.txt').read_text().splitlines()
         if x.strip() and not x.lstrip().startswith('#') and not x.strip().startswith('vllm==')]
    required=requirements_without_cli(metadata.requires('vllm') or [])
    pip('install',*top,*required,'fastapi==0.115.12')
    # Preserve all non-CLI FastAPI standard dependencies, e.g. uvicorn,
    # python-multipart and email-validator, needed by the actual API server.
    standard=requirements_without_cli(metadata.requires('fastapi') or [],extra='standard')
    pip('install',*standard)
    pip('check')
    print('Pinned runtime installed; unused FastAPI CLI omitted; native WebShop scoring unchanged.')


if __name__=='__main__':main()
