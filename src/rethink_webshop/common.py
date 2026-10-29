"""Project-relative paths and small I/O helpers; importing never starts a job."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / '.deps/sdar'
WEBSHOP = UPSTREAM / 'agent_system/environments/env_package/webshop/webshop'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((ROOT/'configs'/name).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def confined(path):
    result = (ROOT/path).resolve()
    if result == ROOT or not result.is_relative_to(ROOT):
        raise ValueError('Use a non-root project-relative path')
    return result
