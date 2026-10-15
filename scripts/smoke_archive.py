"""Load the built archive through Kaggle's file-agent loader in an isolated process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from common import write_json

CHILD = r'''
import json, pathlib, sys
from kaggle_environments import make
root=pathlib.Path(sys.argv[1]).resolve()
env=make('kaggriculture',configuration={'seed':int(sys.argv[2])},debug=True)
env.run([str(root/'main.py'),str(root/'main.py')])
summary=dict(seed=int(sys.argv[2]),frames=len(env.steps),status=[s.status for s in env.state],
             cash=[s.reward for s in env.state])
assert summary['status']==['DONE','DONE'],summary
assert summary['frames']==720,summary
for name,module in list(sys.modules.items()):
    if name.startswith(('market_game_20260930','runtime.','node_rl.','mining.','corpus.','tools.')) or name=='compact_agent':
        file=getattr(module,'__file__',None)
        if file: assert pathlib.Path(file).resolve().is_relative_to(root),(name,file)
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary),encoding='utf-8')
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',default='results/local/archive_smoke.json')
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='kaggriculture-release-') as temp:
        root=Path(temp)
        with tarfile.open(args.archive,'r:gz') as tar:
            for member in tar.getmembers():
                target=(root/member.name).resolve()
                if not target.is_relative_to(root.resolve()) or not member.isfile():
                    raise ValueError('Unsafe archive member')
            tar.extractall(root,filter='data')
        result=root/'result.json'
        subprocess.run([sys.executable,'-I','-c',CHILD,str(root),str(args.seed),str(result)],
                       cwd=root,check=True,timeout=600)
        summary=json.loads(result.read_text(encoding='utf-8'))
    write_json(args.out,summary); print(summary)

if __name__=='__main__': main()
