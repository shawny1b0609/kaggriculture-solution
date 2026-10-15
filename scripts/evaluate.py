"""Official-engine self-play, with independent per-seat agent state."""
import argparse
from collections import Counter
import time
from common import actor_input, assert_engine, load_entry, write_json

def play(seed):
    from kaggle_environments import make
    assert_engine()
    entry = load_entry()
    env = make('kaggriculture', configuration={'seed':seed}, debug=True)
    env.reset()
    began = time.perf_counter()
    max_seconds = 0.
    for step in range(719):
        actions = []
        for seat in (0, 1):
            started = time.perf_counter()
            action = entry.agent(*actor_input(env, seat))
            max_seconds = max(max_seconds, time.perf_counter()-started)
            assert len(action.get('market', [])) <= 10
            actions.append(action)
        env.step(actions)
        assert env.done == (step == 718), (step, [s.status for s in env.state])
    actors = [entry._agents[s] for s in (0,1)]
    result = dict(seed=seed, frames=len(env.steps), status=[s.status for s in env.state],
                  cash=[f['money'] for f in env.state[0].observation.farms],
                  production_matches=[Counter(a.actual_units)==Counter(a.source_units) for a in actors],
                  actual_production=[dict(a.actual_units) for a in actors],
                  planned_production=[dict(a.source_units) for a in actors],
                  crop_decisions=[a.crop_decisions for a in actors],
                  livestock_decisions=[a.pasture_decisions for a in actors],
                  final_shed=[dict(s.observation.private.shed) for s in env.state],
                  max_action_seconds=max_seconds, seconds=time.perf_counter()-began)
    assert result['status']==['DONE','DONE'], result['status']
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--out', default='results/local/selfplay.json')
    args=p.parse_args()
    rows=[]
    for seed in args.seeds:
        row=play(seed); rows.append(row)
        print({k:row[k] for k in ('seed','cash','production_matches','seconds')}, flush=True)
        write_json(args.out, {'mode':'native_selfplay', 'engine':'1.32.7', 'games':rows})

if __name__=='__main__': main()
