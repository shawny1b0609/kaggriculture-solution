"""Replay legal observations through the same online ledger; fills are labels only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import orjson

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'rule_first_20260930')]
from audit_replays import read_raw
from market_game_20260930.g2.features import PublicTracker,public_features,BINS,NAMES,PRODUCTS

OUT=Path(__file__).resolve().parent


def load_labels(split):
    stem='sample200' if split=='train' else 'validation_sample100'
    path=ROOT/'rule_first_20260930/artifacts'/f'actual_market_{stem}.jsonl.gz'
    labels={};dates={}
    with gzip.open(path,'rb') as stream:
        for line in stream:
            row=orjson.loads(line);eid=row['episode_id'];dates[eid]=row['date']
            if eid not in labels:
                labels[eid]=np.zeros((719,2,len(PRODUCTS)),dtype=np.int16)
            if row['op']=='SELL':
                labels[eid][row['step'],row['seat'],PRODUCTS.index(row['product'])]+=1
    return labels,dates,path


def extract(job):
    eid,date,sales,split,stride=job
    began=time.perf_counter()
    try:
        replay=orjson.loads(read_raw(dict(episode_id=eid,date=date)))
        assert len(replay['steps'])==720
        cfg=replay['configuration']
        trackers=[PublicTracker(),PublicTracker()]
        csum=np.concatenate([np.zeros_like(sales[:1]),sales.cumsum(axis=0)],axis=0)
        xs=[];ys=[];meta=[];heuristic=[]
        for t in range(696):
            for seat in (0,1):
                obs=replay['steps'][t][seat]['observation']
                assert int(obs['player'])==seat
                tracker=trackers[seat]
                tracker.update(obs)
                if (t+seat+int(eid))%stride==0:
                    features=public_features(obs,tracker)
                    for i,p in enumerate(PRODUCTS):
                        xs.append(features[i])
                        ys.append([int(csum[t+b,1-seat,i]-csum[t+a,1-seat,i]) for a,b in BINS])
                        meta.append((t,seat,i))
                        pressure=tracker.opponent_pressure.get(p,0)
                        vis=features[i][10]*100
                        heuristic.append([max(0,pressure*(.75**a-.75**b)/.25+vis*(b-a)/72) for a,b in BINS])
                action=replay['steps'][t+1][seat]['action']
                tracker.record_action(obs,action,cfg)
        target=OUT/'data'/split/f'{date}_{eid}.npz'
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,X=np.asarray(xs,dtype=np.float32),y=np.asarray(ys,dtype=np.float32),
                            baseline=np.asarray(heuristic,dtype=np.float32),meta=np.asarray(meta,dtype=np.int16))
        return dict(episode_id=eid,date=date,split=split,rows=len(xs),path=str(target),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=time.perf_counter()-began,
                    ledger_stats=[dict(t.ledger.stats) for t in trackers])
    except Exception as exc:
        return dict(episode_id=eid,date=date,split=split,error=f'{type(exc).__name__}: {exc}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=int,default=200)
    parser.add_argument('--validation',type=int,default=99)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stride',type=int,default=2)
    args=parser.parse_args()
    jobs=[];sources={}
    for split,limit in (('train',args.train),('validation',args.validation)):
        labels,dates,path=load_labels(split)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        # Labels already have an episode-level time split; not shuffled steps.
        for eid in sorted(labels,key=lambda eid:(dates[eid],int(eid)),reverse=True)[:limit]:
            jobs.append((eid,dates[eid],labels[eid],split,args.stride))
    report=dict(features=NAMES,bins=BINS,stride=args.stride,sources=sources,rows=[])
    dest=OUT/'data_manifest.json'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(extract,j) for j in jobs]):
            report['rows'].append(f.result())
            dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'done':len(report['rows']),'total':len(jobs),'errors':sum('error' in r for r in report['rows'])}),flush=True)
    if any('error' in r for r in report['rows']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
