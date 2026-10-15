"""Build causal public features and executed-sale labels from complete replays."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from common import assert_engine, write_json
from evaluation.contract_core import SourceTransactions
from evaluation import scenario_engine as rules
from kaggle_environments.envs.kaggriculture import kaggriculture as engine
from runtime.source_tape import effective_field_action
from market_game_20260930.g2.features import PublicTracker, public_features, BINS, NAMES, PRODUCTS

def read_replay(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f: return json.load(f)

def executed_sales(replay):
    cfg=NS(**replay['configuration'])
    sales=np.zeros((719,2,len(PRODUCTS)),dtype=np.int32)
    assert len(replay['steps'])==720, 'Use complete 720-frame replays'
    for t in range(719):
        obs=[replay['steps'][t][s]['observation'] for s in (0,1)]
        actions=[replay['steps'][t+1][s]['action'] for s in (0,1)]
        farms=copy.deepcopy(obs[0]['farms']); market=copy.deepcopy(obs[0]['market'])
        private=[copy.deepcopy(o['private']) for o in obs]
        for s in (0,1):
            action=effective_field_action(obs[s],s,actions[s],turns_per_day=24,shed_capacity=cfg.shedCapacity)
            for i,unit in enumerate([action['farmer'],*action['hands']]):
                engine._apply_unit_action(farms[s],private[s],i,unit,cfg.boardSize,t//24,24,cfg.shedCapacity)
        state=[NS(observation=NS(farms=farms,private=private[s],market=market),action=actions[s]) for s in (0,1)]
        with SourceTransactions(state) as ledger: rules._process_market(state,NS(configuration=cfg))
        for s in (0,1):
            post=replay['steps'][t+1][s]['observation']
            assert farms[s]['money']==post['farms'][s]['money'], ('cash alignment',t,s)
            for i,order in enumerate(actions[s].get('market',[])[:10]):
                if order[0]=='SELL' and order[1] in PRODUCTS:
                    sales[t,s,PRODUCTS.index(order[1])]+=ledger.fills.get((s,i),0)
    return sales

def build(replay,sales,eid,stride):
    cfg=replay['configuration']; trackers=[PublicTracker(),PublicTracker()]
    csum=np.concatenate([np.zeros_like(sales[:1]),sales.cumsum(axis=0)],axis=0)
    xs=[]; ys=[]; meta=[]; baseline=[]
    for t in range(696):
        for seat in (0,1):
            obs=replay['steps'][t][seat]['observation']; tracker=trackers[seat]
            tracker.update(obs)
            if (t+seat+int(eid))%stride==0:
                features=public_features(obs,tracker)
                for i,product in enumerate(PRODUCTS):
                    xs.append(features[i])
                    ys.append([int(csum[t+b,1-seat,i]-csum[t+a,1-seat,i]) for a,b in BINS])
                    meta.append((t,seat,i))
                    pressure=tracker.opponent_pressure.get(product,0); vis=features[i][10]*100
                    baseline.append([max(0,pressure*(.75**a-.75**b)/.25+vis*(b-a)/72) for a,b in BINS])
            tracker.record_action(obs,replay['steps'][t+1][seat]['action'],cfg)
    return dict(X=np.asarray(xs,np.float32),y=np.asarray(ys,np.float32),
                baseline=np.asarray(baseline,np.float32),meta=np.asarray(meta,np.int16))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replays',type=Path,required=True,help='Manifest: rows with episode_id, date, split, replay')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--stride',type=int,default=2)
    args=p.parse_args(); assert_engine()
    if args.stride<1: p.error('--stride must be positive')
    rows=json.loads(args.replays.read_text(encoding='utf-8'))['rows']
    ids=[str(r['episode_id']) for r in rows]
    assert len(ids)==len(set(ids)), 'Duplicate episode, including across splits'
    assert all(r['split'] in ('train','validation') for r in rows)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if (args.out_dir/'data_manifest.json').exists(): raise FileExistsError('Existing output manifest')
    manifest=dict(features=NAMES,bins=BINS,stride=args.stride,rows=[])
    for row in rows:
        source=args.replays.parent/row['replay']; replay=read_replay(source)
        data=build(replay,executed_sales(replay),row['episode_id'],args.stride)
        target=args.out_dir/f"{row['episode_id']}.npz"
        if target.exists(): raise FileExistsError(target)
        np.savez_compressed(target,**data)
        manifest['rows'].append(dict(episode_id=str(row['episode_id']),date=row['date'],split=row['split'],
            path=target.name,rows=len(data['X']),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            replay_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        write_json(args.out_dir/'data_manifest.json',manifest)
        print({'episodes':len(manifest['rows']),'rows':len(data['X'])},flush=True)

if __name__=='__main__': main()
