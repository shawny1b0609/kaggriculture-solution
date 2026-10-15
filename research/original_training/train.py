"""Train on whole training episodes; only report, never tune on date-out set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_poisson_deviance
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from market_game_20260930.g2.model import FlowModel

OUT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    source=OUT/'data_manifest.json'
    manifest=json.loads(source.read_text(encoding='utf-8'))
    assert not any('error' in r for r in manifest['rows'])
    train_ids={r['episode_id'] for r in manifest['rows'] if r['split']=='train'}
    val_ids={r['episode_id'] for r in manifest['rows'] if r['split']=='validation'}
    assert train_ids.isdisjoint(val_ids)
    datasets={}
    for split in ('train','validation'):
        rows=[r for r in manifest['rows'] if r['split']==split]
        blocks=[]
        for r in rows:
            assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
            blocks.append(np.load(r['path']))
        datasets[split]=[np.concatenate([b[key] for b in blocks]) for key in ('X','y','baseline')]
        print(split,len(rows),datasets[split][0].shape,flush=True)
    X,y,_=datasets['train'];V,target,baseline=datasets['validation']
    payload=dict(version='causal_public_flow_poisson_v1',features=manifest['features'],bins=manifest['bins'],heads=[],
                 training_episodes=sorted(train_ids),validation_episodes=sorted(val_ids),
                 manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    report=dict(train_episodes=len(train_ids),validation_episodes=len(val_ids),train_rows=len(X),validation_rows=len(V),heads=[])
    export=OUT/'opponent_flow_v1.json'
    began=time.perf_counter()
    with threadpool_limits(limits=args.threads):
        for i,interval in enumerate(manifest['bins']):
            model=HistGradientBoostingRegressor(loss='poisson',max_iter=80,max_leaf_nodes=15,max_depth=4,
                min_samples_leaf=200,l2_regularization=10.,learning_rate=.08,early_stopping=False,random_state=908330000+i)
            model.fit(X,y[:,i])
            prediction=model.predict(V)
            head=dict(base=float(model._baseline_prediction[0,0]),trees=[])
            for predictors in model._predictors:
                nodes=predictors[0].nodes
                head['trees'].append([[-1 if n['is_leaf'] else int(n['feature_idx']),float(n['num_threshold']),
                    int(n['left']),int(n['right']),float(n['value']),bool(n['missing_go_to_left'])] for n in nodes])
            payload['heads'].append(head)
            export.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            portable=FlowModel(export)
            sample=np.linspace(0,len(V)-1,min(1500,len(V)),dtype=int)
            error=max(abs(portable.predict_one(V[j])[i]-prediction[j]) for j in sample)
            assert error<1e-7,error
            scores={'bin':interval,'mean_target':float(target[:,i].mean()),'mean_prediction':float(prediction.mean()),'export_max_error':error}
            for name,p in [('model',prediction),('old_flow_heuristic',baseline[:,i])]:
                scores[name]=dict(mae=float(mean_absolute_error(target[:,i],p)),
                                  rmse=float(mean_squared_error(target[:,i],p)**.5),
                                  poisson_deviance=float(mean_poisson_deviance(target[:,i],np.maximum(p,1e-5))))
            report['heads'].append(scores)
            print(json.dumps(scores),flush=True)
    report.update(seconds=time.perf_counter()-began,model_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                  caution='Prediction only, not a measured terminal cash-margin improvement.')
    (OUT/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
