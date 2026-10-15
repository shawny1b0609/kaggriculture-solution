"""Build a submission tarball from the curated runtime, excluding caches."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from common import ROOT, AGENT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'dist/submission.tar.gz')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    entries={p.relative_to(AGENT).as_posix():p for p in AGENT.rglob('*')
             if p.is_file() and p.suffix in ('.py','.json','.gz') and '__pycache__' not in p.parts}
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md','licenses/Apache-2.0.txt'):
        entries[name]=ROOT/name
    manifest={'files':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(entries.items())}}
    with args.out.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for name,path in sorted(entries.items()):
                content=path.read_bytes(); info=tarfile.TarInfo(name)
                info.size=len(content); info.mode=0o644
                tar.addfile(info,io.BytesIO(content))
            content=json.dumps(manifest,sort_keys=True,indent=2).encode()
            info=tarfile.TarInfo('public_manifest.json'); info.size=len(content); info.mode=0o644
            tar.addfile(info,io.BytesIO(content))
    print(json.dumps({'file':str(args.out),'files':len(entries)+1,
                      'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))

if __name__=='__main__': main()
