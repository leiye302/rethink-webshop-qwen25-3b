"""Resolve the documented internal Conda Java11 layout, not an author path."""
import argparse
import os
from pathlib import Path
import subprocess
from .common import ROOT


def prepare():
    prefix=ROOT/'.venv-opsd'
    candidates=[prefix/'lib/jvm',prefix]
    if (prefix/'lib/jvm').is_dir():
        candidates.extend(sorted((prefix/'lib/jvm').iterdir()))
    target=next((x.resolve() for x in candidates if (x/'bin/java').is_file()
                 and (x/'lib/server/libjvm.so').is_file()),None)
    if target is None or not target.is_relative_to(ROOT):
        raise RuntimeError('Install OpenJDK11 in the documented .venv-opsd Conda prefix')
    check=subprocess.run([str(target/'bin/java'),'-version'],capture_output=True,text=True,check=True)
    if 'version "11.' not in check.stderr+check.stdout:
        raise RuntimeError('The native WebShop runtime requires Java11')
    link=ROOT/'assets/java';link.parent.mkdir(parents=True,exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve()!=target:
            raise RuntimeError('Existing Java link differs; inspect it before changing')
    else:
        link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
    print('Internal Java11 home and JVM library verified.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true',required=True)
    p.parse_args();prepare()
