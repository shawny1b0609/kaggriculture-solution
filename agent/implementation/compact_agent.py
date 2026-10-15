"""Small preset executor with a transparent, untrained public-market selector.

Only the macro selector is new. Official observations, hourly execution,
procurement, weed repair and the seller keep their existing implementation.
The score is an approximate opportunity ranking, NOT a cash forecast/oracle.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent)]

from kaggle_environments.envs.kaggriculture import kaggriculture as engine
from config import SELLABLE
from node_rl.affordability import next_day_contract
from tools.portable_tree_agent import PortableTreeAgent
from tools.production_market_features import observation_context, next_day_input_cost

SELECTOR_VERSION = 'compact_public_margin_rule_v1'


def make_cards(tree):
    """Known own plans; do not read source market prices, sales or final money."""
    cards = {}
    for rid, route in tree.routes.items():
        days = []
        for day in range(30):
            loc = route['edges'][min(day, 28)]['example']
            tape = tree.tapes.day_tape(loc, day)
            contract = next_day_contract(tape)
            seeds, animals, consumed = Counter(), Counter(), Counter()
            for row in tape:
                actions = [row['farmer'], *row['hands']]
                for unit, event in zip(actions, row['unit_events']):
                    if unit[0] == 'PLANT':
                        seeds[unit[1]] += 1
                    elif unit[0] == 'FEED':
                        consumed['WHEAT'] += 1
                    elif unit[0] == 'FERTILIZE':
                        consumed['FERTILIZER'] += 1
                    elif unit[0] == 'PLACE' and unit[1] in engine.ANIMALS:
                        # A warehouse return is not a newly placed animal.
                        if int(event['shed'].get(unit[1], 0)) <= 0:
                            animals[unit[1]] += 1
            fixed = (contract['hire_cost'] + contract['land_cost']
                     + sum(n*engine.CROPS[p]['seed'] for p,n in seeds.items())
                     + sum(n*engine.ANIMALS[p]['cost'] for p,n in animals.items()))
            days.append(dict(production=route['daily_production'][day],
                             consumed=dict(consumed), seeds=dict(seeds),
                             animals=dict(animals), fixed_cost=fixed))
        cards[rid] = days
    return cards


def tail_score(cards, rid, start, ctx):
    """Same remaining horizon for all candidates, so investment isn't ignored.

Unknown future shop openings and rival actions are not supplied. Current
prices are shrunk toward base prices; known shop demand only adds a bounded
premium. Seed/animal/labor/land costs and self-consumed feed/fertilizer are
deducted. Actual timing, price impact and liquidity are left to live tests.
"""
    days = cards[rid][start:]
    net_total = Counter()
    for row in days:
        net_total.update(row['production'])
        net_total.subtract(row['consumed'])
    prices = {}
    for p in SELLABLE:
        base = float(engine.MARKET_PARAMS[p]['base'])
        spot = max(.2*base, min(2*base, float(ctx['market_prices'].get(p, base))))
        average = max(0., net_total[p] / max(1, len(days)))
        demand = max(0., float(ctx['demand'].get(p, 0)))
        premium = .8 + .4 * demand / (1 + demand + average)
        prices[p] = (.4*base + .6*spot) * premium
    score = 0.
    for offset, row in enumerate(days):
        revenue = 0.
        for p in SELLABLE:
            net = row['production'].get(p, 0) - row['consumed'].get(p, 0)
            price = prices[p] if net >= 0 else max(1., float(ctx['market_prices'].get(p, engine.MARKET_PARAMS[p]['base'])))
            revenue += net * price
        score += .98**offset * (revenue - row['fixed_cost'])
    return score


class CompactRuleAgent(PortableTreeAgent):
    def __init__(self, tree, *, seed=0, mode='rule', cards=None):
        super().__init__(tree, None, sample=False, actor_seed=seed, runtime_repairs=True)
        if mode not in ('rule', 'random'):
            raise ValueError(mode)
        self.mode = mode
        self.cards = cards if cards is not None else make_cards(tree)

    def _select_edge(self, obs, edges):
        prefix = tuple((int(r['dst']), str(r['loc'])) for r in self.choice['path'] if r.get('dst') is not None)
        nid = self.tree.prefix_ids.get(prefix)
        nodes = [nid] if nid is not None else []
        if nid is None or int(self.tree.tree[nid]['source_node']) != int(self.node_id):
            nodes = self.tree.by_source.get(int(self.node_id), [])
            self.choice['prefix_recovery'].append(dict(day=len(prefix), node=int(self.node_id), prefixes=len(nodes)))
        available = {(int(e['src']), int(e['dst']), str(e['example'])):e for e in edges}
        allowed = {}
        for node in nodes:
            for option in self.tree.rules.get(node, []):
                key = int(option['src']), int(option['dst']), str(option['example'])
                if key in available:
                    allowed.setdefault(key, (available[key], option))
        candidates = list(allowed.values())
        if not candidates:
            self.choice.setdefault('node_errors', []).append(dict(day=len(prefix), reason='no_compact_suffix'))
            return None
        chosen = candidates[0][0]
        if len(candidates) > 1:
            ctx = observation_context(obs)
            start = int(self.tree.net.nodes[int(self.node_id)]['day'])
            scores, details = [], []
            params = engine._resolve_market_params((obs.get('market') or {}).get('params'))
            for _, option in candidates:
                leaves = self.tree.tree[int(option['child'])]['routes']
                values = [tail_score(self.cards, int(rid), start, ctx) for rid in leaves]
                bill = next_day_input_cost(option, ctx, params)
                # Soft caution only: the entire next-day bill is NOT due now.
                penalty = .2 * max(0., bill - ctx['cash'])
                value = sum(values)/len(values) - penalty
                scores.append(value)
                details.append(dict(routes=leaves, score=value, tail_scores=values,
                                    next_day_bill=bill, soft_cash_penalty=penalty))
            pick = (self.actor_rng.randrange(len(candidates)) if self.mode == 'random'
                    else max(range(len(scores)), key=scores.__getitem__))
            chosen = candidates[pick][0]
            self.decisions.append(dict(effective_day=start, observed_day=int(obs.get('day', 0)),
                                       observed_hour=int(obs.get('hour', 0)), node=int(self.node_id),
                                       action=pick, dst=int(chosen['dst']), options=details,
                                       cash=ctx['cash'], known_demand=ctx['demand'],
                                       prices=ctx['market_prices'], version=SELECTOR_VERSION))
        if not self._opened:
            self._opened = True
            self.root_id = int(chosen['dst'])
            self.choice['root'] = self.root_id
        return chosen
