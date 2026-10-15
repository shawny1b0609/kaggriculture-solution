"""Portable source-flow compiler adapted from the audited historical compiler."""
from collections import Counter, defaultdict
import copy
from types import SimpleNamespace as NS
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as native
from corpus.extract import _action, _obs
from runtime.source_tape import effective_field_action
from evaluation import scenario_engine as rules
from evaluation.contract_core import clean, SourceTransactions

VERSION = 'fixed_source_flows_credit_probe_v2'

def scenario_from_replay(replay, seat):
    # Complete only configuration defaults, never source future observations.
    cfg = dict(make('kaggriculture', configuration=replay['configuration']).configuration)
    events = []
    previous = []
    for frame in replay['steps']:
        obs = frame[0]['observation']
        current = obs['town']['unlocked_shops']
        if current[:len(previous)] != previous:
            raise ValueError('Shop sequence is not append-only')
        for shop in current[len(previous):]:
            events.append(dict(day=int(obs['day']), shop=shop))
        previous = list(current)
    seed = replay.get('info', {}).get('seed', replay.get('configuration', {}).get('seed', 0))
    return dict(configuration=cfg, winner_seat=seat, shop_events=events,
                source_seed=int(seed or 0))

def compile_flows(replay, seat, source_id):
    raw = replay["steps"]
    scenario = scenario_from_replay(replay, seat)
    loc = source_id
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
    scenario['source_product_totals'] = clean(gross)
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
                                   gross_reconstructed=True, night_delivery='observed_and_bag_capacity_checked',
                                   max_protected_inventory=maximum),
                source_flow_totals={p: sum(e['quantity'] for e in events if e['product']==p)
                                    for p in {e['product'] for e in events}})
    return data
