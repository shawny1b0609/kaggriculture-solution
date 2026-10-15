"""Recorded production against a live agent; debt and all costs remain explicit."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time
from common import assert_engine, actor_input, load_entry, write_json
from prepare_flow_data import read_replay
from evaluation.compile_contract import compile_flows
from evaluation.contract_core import FixedSupply
from evaluation import scenario_engine as rules
from kaggle_environments import make

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--replay',type=Path)
    source.add_argument('--contract',type=Path)
    p.add_argument('--source-seat',type=int,choices=[0,1],help='Default: higher final cash')
    p.add_argument('--seed',type=int,default=None,help='Override weed seed, retain source shop sequence')
    p.add_argument('--save-contract',type=Path)
    p.add_argument('--out',default='results/local/contract.json')
    args=p.parse_args(); assert_engine()
    if args.replay:
        replay=read_replay(args.replay)
        seat=args.source_seat
        if seat is None:
            seat=max((0,1),key=lambda s:replay['steps'][-1][s]['observation']['farms'][s]['money'])
        data=compile_flows(replay,seat,args.replay.stem)
    else: data=read_replay(args.contract)
    if args.save_contract:
        args.save_contract.parent.mkdir(parents=True,exist_ok=True)
        if args.save_contract.exists(): raise FileExistsError(args.save_contract)
        with gzip.open(args.save_contract,'wt',encoding='utf-8') as f: json.dump(data,f)
    scenario=data['scenario']; seed=args.seed if args.seed is not None else scenario['source_seed']
    env=make('kaggriculture',configuration={**scenario['configuration'],'seed':seed},debug=True)
    env.reset(); env.interpreter=rules.interpreter; env.scenario_shop_events=scenario['shop_events']
    opp=scenario['winner_seat']; ours=1-opp; entry=load_entry(); began=time.perf_counter()
    with FixedSupply(data,env) as counterparty:
        for t in range(719):
            counterparty.step=t
            actions=[None,None]
            actions[ours]=entry.agent(*actor_input(env,ours))
            opp_obs,cfg=actor_input(env,opp)
            actions[opp]=counterparty.action(opp_obs)
            assert all(len(a['market'])<=10 for a in actions)
            env.step(actions); counterparty.active_state=None
            counterparty.replace_farm(data['rows'][t]['next_farm'])
            assert all(counterparty.private['shed'].get(p,0)>=q for p,q in counterparty.reserve.items())
            assert env.done==(t==718)
        money=[float(f['money']) for f in env.state[0].observation.farms]
        net=money[opp]-counterparty.debt; accounts=counterparty.records
        for s in (0,1):
            residual=counterparty.initial_money[s]+accounts[s]['sales']-accounts[s]['costs']-(net if s==opp else money[s])
            assert abs(residual)<1e-7,('cash accounting',s,residual)
        assert not +counterparty.reserve
        for product,incoming in data['source_flow_totals'].items():
            assert incoming-accounts[opp]['sold'][product]-counterparty.losses[product]==counterparty.private['shed'].get(product,0),product
        actor=entry._agents[ours]
        result=dict(mode='fixed_production_contract',opponent_seller='baseline',seed=seed,
            source_seat=opp,frames=len(env.steps),our_cash=money[ours],opponent_cash=money[opp],
            opponent_debt=counterparty.debt,opponent_net_cash=net,margin=money[ours]-net,
            borrowed_total=counterparty.borrowed,capacity_losses=dict(counterparty.losses),
            production_matches=Counter(actor.actual_units)==Counter(actor.source_units),
            actual_production=dict(actor.actual_units),planned_production=dict(actor.source_units),
            cash_accounting_passed=True,stock_accounting_passed=True,
            source_checks=data['source_checks'],seconds=time.perf_counter()-began)
    write_json(args.out,result); print(result)

if __name__=='__main__': main()
