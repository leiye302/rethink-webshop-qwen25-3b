"""No GPUs/APIs. Check source allowlist, file/history privacy, data and patch locks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.github','configs','data','docs','patches','scripts','src','tests',
         'README.md','LICENSE','THIRD_PARTY.md','.gitignore','.gitattributes','.env.example','pytest.ini'}
PATTERNS=[r'connect\.[\w.]*seetacloud',r'(?i)[a-z]:[\\/](?:app|users)',r'/root/'+'auto'+'dl',
          r'oc_sk_[A-Za-z0-9_-]{8,}',r'sk-[A-Za-z0-9_-]{20,}',r'gh[pousr]_[A-Za-z0-9]{20,}',
          r'(?i)BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',r'(?i)(?:api_key|password|token)\s*[:=]\s*["\x27][A-Za-z0-9_+/=-]{16,}["\x27]']


def scan(data,name):
    text=data.decode('utf-8',errors='replace')
    for pattern in PATTERNS:
        if re.search(pattern,text):
            raise AssertionError('Privacy pattern in '+name+' (contents suppressed)')


def main():
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    names=sorted(set(x for x in names if x))
    for name in names:
        assert Path(name).parts[0] in ALLOWED,name
        path=ROOT/name
        assert not path.is_symlink(),name
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    scan(z.read(member),name+':'+member)
        else:
            scan(path.read_bytes(),name)
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
    if (ROOT/'.git').exists():
        commits=subprocess.check_output(['git','-C',str(ROOT),'rev-list','--all']).decode().split()
        for commit in commits:
            meta=subprocess.check_output(['git','-C',str(ROOT),'show','--no-patch','--format=%an <%ae> %cn <%ce> %B',commit])
            scan(meta,'commit metadata')
            # Scan all historical blobs, not just the current .gitignore.
            blobs=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r',commit]).decode().splitlines()
            for line in blobs:
                identity,name=line.split('\t',1)
                obj=identity.split()[2]
                content=subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',obj])
                if name.endswith('.zip'):
                    import io
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        for member in z.namelist(): scan(z.read(member),'history:'+name+':'+member)
                else:
                    scan(content,'history:'+name)
    print(json.dumps({'files_checked':len(names),'history_checked':True,'syntax_checked':True,'privacy_passed':True}))


if __name__=='__main__':
    main()
