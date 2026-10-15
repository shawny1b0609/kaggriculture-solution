"""Isolated fixed-production opponent probe. Not connected to either PPO trainer.

Compile *realized* source warehouse transfers and purchases once. During a
challenge there is no opponent worker replay. Its production inputs are
reserved, its cash shortfalls are loans, and its seller trades in the live
shared market. This is an explicitly idealized counterparty, not native play.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent), str(OUT)]

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as native
import bc_rl_v2.env.kaggriculture_scenario as rules
from arena.native import actor_input, assert_engine, plain
from config import SELLABLE
from corpus.extract import _action, _obs
from runtime.source_tape import TapeBank, effective_field_action
from tools.dedup_grid_root_daily import source_jobs
from tools.reliable_market_seller import ReliableMarketSeller
from tools import diagnose_route_ceiling as diagnostic
from tools.portable_tree_agent import PortableTreeAgent
from tools.production_market_features import OptionNet
import audit

VERSION = 'fixed_source_flows_credit_probe_v2'
DEST = OUT / 'fixed_supply_probe'


def clean(counter):
    return {str(k): int(v) for k, v in counter.items() if int(v)}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=3) as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


class SourceTransactions:
    """Observe successful source transactions, including partial order fills."""
    def __init__(self, state):
        self.farms = {id(f): s for s, f in enumerate(state[0].observation.farms)}
        self.orders = {id(o): (s, i) for s, st in enumerate(state)
                       for i, o in enumerate(st.action.get('market', [])[:10])}
        self.active = {}
        self.fills = defaultdict(int)
        self.costs = Counter()
        self.originals = {}

    def __enter__(self):
        for name in ('_parse_order', '_commit_unit', '_do_hire', '_do_buy_land'):
            self.originals[name] = getattr(rules, name)
        def parse(order):
            if id(order) in self.orders:
                s, i = self.orders[id(order)]
                self.active[s] = i
            return self.originals['_parse_order'](order)
        def unit(op, item, price, farm, *args):
            before = farm['money']
            ok = self.originals['_commit_unit'](op, item, price, farm, *args)
            s = self.farms[id(farm)]
            if ok:
                self.fills[s, self.active[s]] += 1
                if op != 'SELL':
                    self.costs[s] += before - farm['money']
            return ok
        def atomic(name, farm, *args):
            before = farm['money']
            result = self.originals[name](farm, *args)
            if farm['money'] != before:
                s = self.farms[id(farm)]
                self.fills[s, self.active[s]] += 1
                self.costs[s] += before - farm['money']
            return result
        rules._parse_order = parse
        rules._commit_unit = unit
        rules._do_hire = lambda f, *a: atomic('_do_hire', f, *a)
        rules._do_buy_land = lambda f, *a: atomic('_do_buy_land', f, *a)
        return self

    def __exit__(self, *args):
        for k, v in self.originals.items():
            setattr(rules, k, v)


def compile_flows(loc):
    path = DEST / ('source_v2_' + loc.replace(':', '_') + '.json.gz')
    if path.exists():
        data = diagnostic.read_gzip(path)
        assert data['version'] == VERSION
        return data
    bank = TapeBank(source_jobs(PROJECT.parent / 'data/featured_index/refresh_20260927_node_rl/manifest.csv'),
                    max_cached_episodes=1)
    raw = bank._load_steps(loc)
    scenario = diagnostic.source_scenario(171223, loc)
    seat = int(loc.split(':')[-1])
    assert len(raw) == 720
    cfg = NS(**scenario['configuration'])
    boardsize, cap = cfg.boardSize, cfg.shedCapacity
    observations = []
    previous = [None, None]
    for frame in raw:
        current = [_obs(frame, s, previous[s]) for s in range(2)]
        if any(o is None or 'private' not in o for o in current):
            raise ValueError('incomplete source private observations')
        observations.append(current)
        previous = current
    events, rows, seed_events = [], [], []
    def event(product, quantity, phase):
        row = dict(product=product, quantity=int(quantity), phase=phase, reserved=0)
        events.append(row)
        return row
    initial = [event(p, q, 'initial') for p, q in observations[0][seat]['private']['shed'].items() if q]
    for p, q in observations[0][seat]['private']['seeds'].items():
        if q:
            seed_events.append(dict(product=p, quantity=q, reserved=0))
    total_cost = 0
    gross = Counter()
    for t in range(719):
        pre, post = observations[t], observations[t+1]
        actions = [_action(raw[t+1], s) or {} for s in range(2)]
        farms = copy.deepcopy(pre[0]['farms'])
        privates = [copy.deepcopy(pre[s]['private']) for s in range(2)]
        row = dict(step=t, field=[], night=[], operations=[], seed_delta={},
                   pre_farm=copy.deepcopy(farms[seat]))
        for s in range(2):
            action, meta = effective_field_action(pre[s], s, actions[s],
                turns_per_day=cfg.turnsPerDay, shed_capacity=cap, include_meta=True)
            for idx, unit in enumerate([action['farmer'], *action['hands']]):
                native._apply_unit_action(farms[s], privates[s], idx, unit, boardsize,
                                         t//24, 24, cap)
            if s == seat:
                row['field_farm'] = copy.deepcopy(farms[s])
                row['seed_delta'] = meta['seed_delta_units']
                for p, q in meta['seed_delta_units'].items():
                    if q:
                        seed_events.append(dict(product=p, quantity=q, reserved=0))
                row['yielded'] = meta['yielded']
                gross.update(meta['yielded'])
                for unit in meta['unit_events']:
                    for p, q in unit['shed'].items():
                        if q:
                            row['field'].append(event(p, q, 'field'))
        state = [NS(observation=NS(farms=farms, private=privates[s],
                                  market=None), action=actions[s]) for s in range(2)]
        market = copy.deepcopy(pre[0]['market'])
        for st in state:
            st.observation.market = market
        with SourceTransactions(state) as ledger:
            rules._process_market(state, NS(configuration=cfg))
        total_cost += ledger.costs[seat]
        for i, order in enumerate(actions[seat].get('market', [])[:10]):
            n = ledger.fills.get((seat, i), 0)
            if not n or order[0] == 'SELL':
                continue
            operation = list(order) if len(order) < 3 else [order[0], order[1], n]
            item = dict(order=operation, flow=None, seed_flow=None)
            if order[0] in ('BUY_PRODUCT', 'BUY_ANIMAL'):
                item['flow'] = event(order[1], n, 'purchase')
            elif order[0] == 'BUY_SEED':
                item['seed_flow'] = dict(product=order[1], quantity=n, reserved=0)
                seed_events.append(item['seed_flow'])
            row['operations'].append(item)
        if (t+1) % 24 == 0:
            for s in range(2):
                before = dict(privates[s]['shed'])
                # At capacity, the original in-memory bag key insertion order
                # need not survive replay serialization. Read the actual night
                # delivery from the two states, and verify bag/capacity bounds.
                carried = Counter()
                for bag in privates[s]['inventories']:
                    carried.update(bag)
                delivered = {p: int(post[s]['private']['shed'].get(p, 0))-int(before.get(p, 0))
                             for p in set(before) | set(post[s]['private']['shed'])}
                assert all(0 <= n <= carried[p] for p, n in delivered.items()), ('night units', t, s)
                assert sum(delivered.values()) == min(sum(carried.values()), max(0, cap-sum(before.values()))), ('night capacity', t, s)
                privates[s]['shed'] = copy.deepcopy(post[s]['private']['shed'])
                privates[s]['inventories'] = [{}]
                if s == seat:
                    for p, q in privates[s]['shed'].items():
                        change = q - before.get(p, 0)
                        if change:
                            row['night'].append(event(p, change, 'night'))
        # Source quantities/costs are verified against the following frame.
        for s in range(2):
            assert farms[s]['money'] == post[s]['farms'][s]['money'], ('cash alignment', t, s)
            for key in ('shed', 'seeds'):
                assert clean(privates[s][key]) == clean(post[s]['private'][key]), (key, t, s)
        row['next_farm'] = copy.deepcopy(post[seat]['farms'][seat])
        rows.append(row)
    # Allocate the latest already-arrived units to each source farm withdrawal.
    # Everything not required for future farm withdrawals is saleable at its
    # original arrival time. Historical SELL commands do not become supply.
    for stream in (events, seed_events):
        stacks = defaultdict(list)
        for e in stream:
            p, q = e['product'], e['quantity']
            if q > 0:
                stacks[p].append([e, q])
            else:
                need = -q
                while need:
                    if not stacks[p]:
                        raise ValueError(f'noncausal source material balance {loc} {p}')
                    parent, remaining = stacks[p][-1]
                    take = min(need, remaining)
                    parent['reserved'] += take
                    need -= take
                    stacks[p][-1][1] -= take
                    if not stacks[p][-1][1]:
                        stacks[p].pop()
    removed = Counter()
    for row in rows:
        required = []
        for item in row['operations']:
            flow = item['flow'] or item['seed_flow']
            if flow:
                removed[item['order'][0]+':'+flow['product']] += flow['quantity']-flow['reserved']
                flow['quantity'] = flow['reserved']
                item['order'][2] = flow['reserved']
                if not flow['reserved']:
                    continue
            required.append(item)
        row['operations'] = required
    assert clean(gross) == clean(scenario['source_product_totals'])
    reserve = Counter()
    maximum = 0
    for e in events:
        reserve[e['product']] += e['reserved'] if e['quantity'] > 0 else e['quantity']
        assert min(reserve.values(), default=0) >= 0
        maximum = max(maximum, sum(reserve.values()))
    assert not +reserve
    assert maximum <= cap, ('protected input capacity', maximum)
    data = dict(version=VERSION, scenario=scenario, rows=rows, initial=initial,
                initial_seeds=observations[0][seat]['private']['seeds'],
                source_operating_cost=total_cost, source_cash=observations[-1][seat]['farms'][seat]['money'],
                omitted_nonproduction_purchases=clean(removed),
                source_checks=dict(transitions=719, seats=2, cash_exact=True,
                                   shed_exact=True, seeds_exact=True,
                                   gross_exact=True, night_delivery='observed_and_bag_capacity_checked',
                                   max_protected_inventory=maximum),
                source_flow_totals={p: sum(e['quantity'] for e in events if e['product']==p)
                                    for p in {e['product'] for e in events}})
    save(path, data)
    return data


class FixedSupply:
    def __init__(self, data, env):
        self.data, self.env = data, env
        self.seat = data['scenario']['winner_seat']
        self.reserve = Counter()
        self.seller = ReliableMarketSeller()
        self.debt = self.borrowed = self.peak_debt = 0.
        self.records = [dict(sales=0., costs=0., sold=Counter(), bought=Counter()) for _ in range(2)]
        self.losses = Counter()
        self.step = 0
        self.originals = {}
        self.buy_queue = []
        self.active_state = None
        self.initial_money = [float(f['money']) for f in env.state[0].observation.farms]

    @property
    def farm(self):
        return (self.active_state if self.active_state is not None else self.env.state)[0].observation.farms[self.seat]

    @property
    def private(self):
        return (self.active_state if self.active_state is not None else self.env.state)[self.seat].observation.private

    def replace_farm(self, source):
        money = self.farm['money']
        self.farm.clear()
        self.farm.update(copy.deepcopy(source))
        self.farm['money'] = money
        self.private['inventories'] = [{} for _ in range(1+len(source['hands']))]

    def deposit(self, private, reserve, e, losses):
        p, q = e['product'], e['quantity']
        shed = private['shed']
        if q < 0:
            assert reserve[p] >= -q and shed.get(p, 0) >= -q, (self.step, p, q, reserve[p], shed.get(p, 0))
            reserve[p] += q
            shed[p] += q
            return
        reserved = e['reserved']
        capacity = int(self.env.configuration.shedCapacity)
        # Guaranteed-production assumption: protect production inputs. If a
        # slower seller fills the shed, surplus goods, never inputs, are lost.
        while sum(shed.values()) + reserved > capacity:
            free = [k for k, n in shed.items() if n > reserve[k]]
            if not free:
                raise ValueError('production reservation exceeds capacity')
            chosen = min(free, key=lambda k: (native.MARKET_PARAMS.get(k, {}).get('base', 0), k))
            shed[chosen] -= 1
            losses[chosen] += 1
        free_room = max(0, capacity - sum(shed.values()) - reserved)
        taken = reserved + min(q-reserved, free_room)
        shed[p] = shed.get(p, 0) + taken
        reserve[p] += reserved
        losses[p] += q-taken

    def field(self, private, reserve, losses):
        row = self.data['rows'][self.step]
        for e in row['field']:
            self.deposit(private, reserve, e, losses)
        for p, q in row['seed_delta'].items():
            private['seeds'][p] = private['seeds'].get(p, 0) + q
            assert private['seeds'][p] >= 0

    def action(self, obs):
        row = self.data['rows'][self.step]
        projected, reserved = copy.deepcopy(self.private), copy.deepcopy(self.reserve)
        self.field(projected, reserved, Counter())
        orders = [item['order'] for item in row['operations']]
        assert len(orders) <= 10
        forecast = Counter()
        for future in self.data['rows'][self.step:min(719, (self.step//24+1)*24)]:
            for event in future['field'] + future['night']:
                if event['quantity'] > 0:
                    forecast[event['product']] += event['quantity'] - event['reserved']
        # Use live prices for the immediate operations. This is a liquidity
        # budget, not a future quotation or source final-money target.
        budget, hires = 0., int(row['field_farm']['hires_today'])
        lands = len(row['field_farm']['unlocked_quadrants']) - 1
        for order in orders:
            op = order[0]
            if op == 'HIRE':
                budget += native._hire_cost(hires, self.env.configuration.farmHandCostMult)
                hires += 1
            elif op == 'BUY_LAND':
                budget += native.LAND_PRICES[lands]
                lands += 1
            elif op == 'BUY_SEED':
                budget += native.CROPS[order[1]]['seed'] * order[2]
            elif op == 'BUY_ANIMAL':
                budget += native.ANIMALS[order[1]]['cost'] * order[2]
            elif op == 'BUY_PRODUCT':
                p, n = order[1:]
                budget += sum(native.market_price(p, obs['market']['inventory'][p]-j-1,
                                                    obs['market'].get('params')) for j in range(n))
        seller_obs = copy.deepcopy(obs)
        seller_obs['farms'][self.seat]['money'] -= self.debt
        plan = self.seller.plan(seller_obs, dict(projected['shed']), dict(reserved),
                               minimum_cash=budget, available_order_slots=10-len(orders),
                               required_capacity=sum(item['flow']['quantity'] for item in row['operations']
                                                     if item['flow']),
                               projected_arrivals=dict(forecast), terminal=self.step>=718)
        return dict(farmer=['PASS'], hands=[], market=plan.orders + orders)

    def borrow_for(self, farm, amount):
        shortage = max(0., amount - farm['money'])
        farm['money'] += shortage
        self.debt += shortage
        self.borrowed += shortage
        self.peak_debt = max(self.peak_debt, self.debt)

    def __enter__(self):
        for key in ('_process_market', '_commit_unit', '_do_hire', '_do_buy_land', '_end_of_day'):
            self.originals[key] = getattr(rules, key)
        def market(state, env):
            # Kaggle passes a copied working state into the interpreter;
            # env.state still denotes the previous frame until env.step ends.
            self.active_state = state
            row = self.data['rows'][self.step]
            self.replace_farm(row['field_farm'])
            self.field(self.private, self.reserve, self.losses)
            self.buy_queue = [[item['order'][1], item['flow']['quantity'], item['flow']['reserved']]
                              for item in row['operations'] if item['flow']]
            self.farm_ids = {id(f): s for s, f in enumerate(state[0].observation.farms)}
            result = self.originals['_process_market'](state, env)
            assert not self.buy_queue, ('incomplete fixed procurement', self.step, self.buy_queue)
            return result
        def commit(op, item, price, farm, private, market, cap=100):
            seat = self.farm_ids[id(farm)]
            protected = 0
            if seat == self.seat and op != 'SELL':
                self.borrow_for(farm, price)
                if op in ('BUY_PRODUCT', 'BUY_ANIMAL'):
                    assert self.buy_queue and self.buy_queue[0][0] == item
                    protected = int(self.buy_queue[0][2] > 0)
                    # Make room only for a guaranteed production purchase;
                    # the displaced sale stock is explicitly recorded as loss.
                    while sum(private['shed'].values()) >= cap:
                        free = [p for p, n in private['shed'].items() if n > self.reserve[p]]
                        if not free:
                            raise ValueError('purchase cannot fit protected warehouse')
                        p = min(free, key=lambda p: (native.MARKET_PARAMS.get(p, {}).get('base', 0), p))
                        private['shed'][p] -= 1
                        self.losses[p] += 1
            before = float(farm['money'])
            ok = self.originals['_commit_unit'](op, item, price, farm, private, market, cap)
            record = self.records[seat]
            if ok:
                if op == 'SELL':
                    record['sales'] += float(farm['money']) - before
                    record['sold'][item] += 1
                    if seat == self.seat:
                        repayment = min(self.debt, farm['money'])
                        farm['money'] -= repayment
                        self.debt -= repayment
                else:
                    record['costs'] += before - float(farm['money'])
                    record['bought'][op+':'+item] += 1
                    if seat == self.seat and op in ('BUY_PRODUCT', 'BUY_ANIMAL'):
                        self.reserve[item] += protected
                        self.buy_queue[0][1] -= 1
                        self.buy_queue[0][2] -= protected
                        if not self.buy_queue[0][1]:
                            self.buy_queue.pop(0)
            elif seat == self.seat and op != 'SELL':
                raise ValueError(('guaranteed purchase rejected', self.step, op, item))
            return ok
        def atomic(key, farm, *args):
            seat = self.farm_ids[id(farm)]
            if seat == self.seat:
                cost = (native._hire_cost(farm['hires_today'], args[-1]) if key == '_do_hire'
                        else native.LAND_PRICES[len(farm['unlocked_quadrants'])-1])
                self.borrow_for(farm, cost)
            before = float(farm['money'])
            value = self.originals[key](farm, *args)
            self.records[seat]['costs'] += before - float(farm['money'])
            return value
        def end_day(state, env, day):
            result = self.originals['_end_of_day'](state, env, day)
            for e in self.data['rows'][self.step]['night']:
                self.deposit(self.private, self.reserve, e, self.losses)
            return result
        rules._process_market = market
        rules._commit_unit = commit
        rules._do_hire = lambda f, *a: atomic('_do_hire', f, *a)
        rules._do_buy_land = lambda f, *a: atomic('_do_buy_land', f, *a)
        rules._end_of_day = end_day
        self.replace_farm(self.data['rows'][0]['pre_farm'])
        self.private['seeds'] = copy.deepcopy(self.data['initial_seeds'])
        for e in self.data['initial']:
            self.deposit(self.private, self.reserve, e, self.losses)
        return self

    def __exit__(self, *args):
        for k, v in self.originals.items():
            setattr(rules, k, v)


def run_probe(root, route, case, fresh=False):
    path = DEST / f'root{root}_route{route}_case{case}.json'
    if path.exists() and not fresh:
        return json.loads(path.read_text(encoding='utf-8'))
    audit.initialize(root)
    loc = audit.CASES[case]
    data = compile_flows(loc)
    scenario = data['scenario']
    tree = copy.copy(diagnostic.TREE)
    tree.rules = {parent: [o for o in options if route in tree.tree[int(o['child'])]['routes']]
                  for parent, options in tree.rules.items()}
    agent = PortableTreeAgent(tree, OptionNet(), sample=False, actor_seed=0, runtime_repairs=True)
    assert_engine()
    env = make('kaggriculture', configuration={**scenario['configuration'], 'seed': scenario['source_seed']}, debug=True)
    env.reset()
    env.interpreter = rules.interpreter
    env.scenario_shop_events = scenario['shop_events']
    opp, ours = scenario['winner_seat'], 1-scenario['winner_seat']
    errors = []
    began = time.perf_counter()
    with FixedSupply(data, env) as counterparty:
        for t in range(719):
            counterparty.step = t
            our_obs, cfg = actor_input(env, ours)
            opp_obs, _ = actor_input(env, opp)
            try:
                our_action = agent(our_obs, cfg)
                assert len(our_action['market']) <= 10
            except Exception as exc:
                errors.append(dict(step=t, exception=repr(exc)))
                our_action = dict(farmer=['PASS'], hands=[], market=[])
            opp_action = counterparty.action(opp_obs)
            assert len(opp_action['market']) <= 10
            actions = [None, None]
            actions[ours], actions[opp] = our_action, opp_action
            env.step(actions)
            counterparty.active_state = None
            counterparty.replace_farm(data['rows'][t]['next_farm'])
            assert all(counterparty.private['shed'].get(p, 0) >= q
                       for p, q in counterparty.reserve.items()), ('reserved inventory', t)
            assert env.done == (t == 718)
        money = [float(f['money']) for f in env.state[0].observation.farms]
        net = money[opp] - counterparty.debt
        accounts = counterparty.records
        for s in range(2):
            actual = net if s == opp else money[s]
            error = counterparty.initial_money[s] + accounts[s]['sales'] - accounts[s]['costs'] - actual
            assert abs(error) < 1e-7, ('cash accounting', s, error)
        assert not +counterparty.reserve
        # The compiled transfers, not their annotated harvest totals, are what
        # enter live stock. Check a second, independent warehouse balance.
        stock_error = {}
        for p, incoming in data['source_flow_totals'].items():
            actual = int(counterparty.private['shed'].get(p, 0))
            residual = incoming - accounts[opp]['sold'][p] - counterparty.losses[p] - actual
            if residual:
                stock_error[p] = residual
        assert not stock_error, stock_error
        result = dict(version=VERSION, root=root, route=route, case=case, loc=loc,
                      our_cash=money[ours], opponent_cash=money[opp], opponent_debt=counterparty.debt,
                      opponent_net_cash=net, cash_delta=money[ours]-net,
                      borrowed_total=counterparty.borrowed, peak_debt=counterparty.peak_debt,
                      source_operating_cost=data['source_operating_cost'], source_cash=data['source_cash'],
                      source_checks=data['source_checks'], accounts=accounts,
                      our_stable=agent.choice['hits']>=30 and sum(agent.actual_units.values())/max(1,sum(agent.source_units.values()))>=.95,
                      our_hits=agent.choice['hits'], our_production=dict(agent.actual_units),
                      opponent_production=scenario['source_product_totals'],
                      opponent_production_mode='fixed_source_flow_contract_not_replayed_worker_actions',
                      opponent_capacity_losses=clean(counterparty.losses), opponent_final_shed=clean(counterparty.private['shed']),
                      our_errors=errors, steps=719, cash_check_error=0, stock_check_error=stock_error,
                      shops=plain(env.state[0].observation.town), seconds=time.perf_counter()-began)
    old_path = OUT/f'root{root}/games/route{route}_case_{case}.json.gz'
    if old_path.exists():
        old = diagnostic.read_gzip(old_path)
        result['old_replay'] = {k: old[k] for k in ('our_cash','opponent_cash','cash_delta','accepted_for_learning','farm_stable')}
    save(path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=int, default=476649)
    parser.add_argument('--route', type=int, default=713)
    parser.add_argument('--case', type=int, default=2)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.compile_only:
        data = compile_flows(audit.CASES[args.case])
        print(json.dumps({k: data[k] for k in ('source_checks','source_operating_cost','source_cash')}, ensure_ascii=False))
    else:
        run_probe(args.root, args.route, args.case, args.fresh)
