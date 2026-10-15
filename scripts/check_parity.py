"""Compare the public runtime against actions in a compatible reference replay."""
import argparse
import gzip
import json
from pathlib import Path
from common import CONFIG_KEYS, assert_engine, load_entry, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay', type=Path)
    p.add_argument('--out', default='results/local/parity.json')
    p.add_argument('--steps', type=int, default=719)
    args=p.parse_args()
    opener=gzip.open if args.replay.suffix=='.gz' else open
    with opener(args.replay,'rt',encoding='utf-8') as f: replay=json.load(f)
    assert_engine(); entry=load_entry(); mismatches=[]; compared=0
    cfg={k:v for k,v in replay['configuration'].items() if k in CONFIG_KEYS}
    for t in range(min(args.steps,len(replay['steps'])-1)):
        for seat in (0,1):
            obs=replay['steps'][t][seat]['observation']
            expected=replay['steps'][t+1][seat]['action']
            actual=entry.agent(obs,cfg)
            compared+=1
            if actual!=expected: mismatches.append({'step':t,'seat':seat})
    write_json(args.out,dict(compared=compared,mismatches=mismatches))
    print({'compared':compared,'mismatches':len(mismatches)})
    if mismatches: raise SystemExit(1)

if __name__=='__main__': main()
