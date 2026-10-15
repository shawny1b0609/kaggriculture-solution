"""Versioned, prefix-constrained policy using the accepted v4 execution layer."""
from __future__ import annotations
from collections import Counter
import copy
import gzip
import json
import math
from pathlib import Path
import random
import torch

from runtime.grid_store import GridNet
from runtime.grid_agent import GridAgent
from runtime.source_tape import effective_field_action
from node_rl.features import context
from node_rl.affordability import affordability_vector
from tools.portable_branch_features import PRODUCTS, option_features, observable_option_score
from tools.compact_branch_features import option_features as compact_option_features, FEATURE_VERSION
from tools.production_market_features import (
    option_features as production_market_option_features, observation_context,
    FEATURE_VERSION as PRODUCTION_MARKET_VERSION,
)
from tools.reliable_market_seller import ReliableMarketSeller
from tools.stable_execution_agent import StableExecutionMixin
from tools.slot_guard_experiment import protect_next_dawn_slots
from tools.capacity_guard_experiment import protect_next_dawn_capacity


class AssetTapes:
    def __init__(self, sources):
        self.sources = sources

    def day_tape(self, loc, day):
        return copy.deepcopy(self.sources[str(loc)]['tapes'].get(str(day)) or [])

    def dawn_farm(self, loc, day):
        return copy.deepcopy(self.sources[str(loc)]['dawns'].get(str(day)))


class PortableTree:
    def __init__(self, path):
        with gzip.open(Path(path), 'rt', encoding='utf-8') as handle:
            asset = json.load(handle)
        supplementary = Path(path).parent / 'witnesses.json.gz'
        if supplementary.exists():
            with gzip.open(supplementary, 'rt', encoding='utf-8') as handle:
                extra = json.load(handle)
            for loc, buckets in extra.items():
                source = asset['sources'].setdefault(loc, {'tapes': {}, 'dawns': {}})
                for bucket in ('tapes', 'dawns'):
                    source[bucket].update(buckets.get(bucket) or {})
        for node in asset['nodes']:
            day, loc = int(node['day']), str(node.get('example') or '')
            if day > 0 and loc:
                for d in (day - 1, day):
                    if not asset['sources'].get(loc, {}).get('dawns', {}).get(str(d)):
                        raise ValueError(f'incomplete canonical witness: {loc} day {d}')
        self.root = int(asset['root'])
        self.routes = {int(r['path_index']): r for r in asset['routes']}
        self.tree = asset['tree']
        self.rules = {int(r['prefix_node']): r['options'] for r in asset['rules']}
        # Distant production potential matters even when two early options
        # share today's output. These are own known plans, never future prices.
        for parent, options in self.rules.items():
            start = int(self.tree[parent]['day'])
            for option in options:
                leaves = self.tree[int(option['child'])]['routes']
                totals = Counter()
                for rid in leaves:
                    for day in self.routes[int(rid)]['daily_production'][start:]:
                        totals.update(day)
                option['remaining_daily'] = {p: totals[p] / (len(leaves) * max(1, 30 - start)) for p in PRODUCTS}
        self.prefix_ids = {(): 0}
        self.by_source = {}
        for node in self.tree:
            self.by_source.setdefault(int(node['source_node']), []).append(int(node['id']))
        for route in self.routes.values():
            cursor, prefix = 0, ()
            for edge in route['edges']:
                child = next(c for c in self.tree[cursor]['children']
                             if int(self.tree[c]['source_node']) == int(edge['dst'])
                             and self.tree[c]['example'] == edge['example'])
                prefix += ((int(edge['dst']), str(edge['example'])),)
                self.prefix_ids[prefix] = child
                cursor = child
        nodes = {int(n['id']): n for n in asset['nodes']}
        outgoing = {}
        for edge in asset['edges']:
            outgoing.setdefault(int(edge['src']), []).append(edge)
        self.net = GridNet(nodes, {(int(n['day']), str(n['grid'])): int(n['id'])
                                   for n in nodes.values()}, outgoing)
        self.tapes = AssetTapes(asset['sources'])
        for parent, options in self.rules.items():
            for option in options:
                start = int(self.tree[parent]['day'])
                cursor = self.tree[int(option['child'])]
                while len(cursor['children']) == 1:
                    cursor = self.tree[int(cursor['children'][0])]
                option['next_choice_options'] = len(cursor['children'])
                # Read-only metadata; do not deep-copy the entire daily tape.
                tape = asset['sources'].get(str(option['example']), {}).get('tapes', {}).get(str(start), [])
                option['peak_farm_orders'] = max(
                    (sum(bool(o) and o[0] != 'SELL' for o in step.get('market') or [])
                     for step in tape), default=0)


class GuardedExecutionAgent(StableExecutionMixin, GridAgent):
    """Same worker/purchase/seller rules for learner and replay opponent."""
    def __init__(self, *args, runtime_repairs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime_repairs = runtime_repairs
        if runtime_repairs:
            self.market_seller = ReliableMarketSeller()

    def __call__(self, obs, cfg=None):
        action = super().__call__(obs, cfg)
        before = [list(o) for o in action.get('market', []) if o and o[0] == 'SELL']
        protect_next_dawn_slots(self, obs, action)
        protect_next_dawn_capacity(self, obs, action, ordered=self.runtime_repairs)
        after = [list(o) for o in action.get('market', []) if o and o[0] == 'SELL']
        if before != after and self.sell_mode == 'seller':
            audit = self.choice.get('market_sell_accounting') or {}
            for key, difference in (
                ('issued_orders', len(after) - len(before)),
                ('issued_units', sum(o[2] for o in after) - sum(o[2] for o in before)),
                ('issued_stressed_revenue_estimate',
                 self.market_seller.stressed_revenue(obs, after) - self.market_seller.stressed_revenue(obs, before))):
                audit[key] = audit.get(key, 0) + difference
            plans = self.choice.get('market_sell_plans') or []
            if plans and (plans[-1].get('day'), plans[-1].get('hour')) == (obs.get('day'), obs.get('hour')):
                plans[-1]['actual_orders'] = after
                plans[-1]['actual_stressed_revenue_estimate'] = round(self.market_seller.stressed_revenue(obs, after), 3)
        return action


class RepairedExecutionAgent(GuardedExecutionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, runtime_repairs=True, **kwargs)


class PortableTreeAgent(GuardedExecutionAgent):
    def __init__(self, tree, model, *, sample, actor_seed, runtime_repairs=False):
        super().__init__(tree.net, root_id=tree.root, tapes=tree.tapes,
                         market_mode='derived', sell_mode='seller', seed=actor_seed,
                         runtime_repairs=runtime_repairs)
        self.tree, self.model, self.sample = tree, model, sample
        self.actor_rng = random.Random(actor_seed)
        self.decisions = []
        self.price_ema, self.flow_ema, self.sales_ema = {}, {}, {}
        self.previous_inventory = None
        self.actual_units, self.source_units = Counter(), Counter()
        self.last_source_day = -1
        self.choice['node_risk'] = []
        self.choice['prefix_recovery'] = []

    def __call__(self, obs, cfg=None):
        market = obs.get('market') or {}
        inventory, prices = market.get('inventory') or {}, market.get('prices') or {}
        for p in PRODUCTS:
            spot = float(prices.get(p) or 0)
            self.price_ema[p] = .95 * self.price_ema.get(p, spot) + .05 * spot
            if self.previous_inventory is not None:
                flow = float(inventory.get(p) or 0) - float(self.previous_inventory.get(p) or 0)
                self.flow_ema[p] = .9 * self.flow_ema.get(p, 0.) + .1 * flow
        self.previous_inventory = dict(inventory)
        action = super().__call__(obs, cfg)
        sales = Counter()
        for order in action.get('market') or []:
            if len(order) >= 3 and order[0] == 'SELL':
                sales[str(order[1])] += int(order[2])
        for p in PRODUCTS:
            self.sales_ema[p] = .9 * self.sales_ema.get(p, 0.) + .1 * sales[p]
        _, meta = effective_field_action(obs, int(obs['player']), action, include_meta=True)
        self.actual_units.update(meta.get('yielded') or {})
        day = int(obs.get('day') or 0)
        if day != self.last_source_day:
            self.last_source_day = day
            if self.loc:
                for step in self.tapes.day_tape(self.loc, day):
                    self.source_units.update(step.get('yielded') or {})
        return action

    def _select_edge(self, obs, edges):
        prefix = tuple((int(r['dst']), str(r['loc'])) for r in self.choice['path'] if r.get('dst') is not None)
        nid = self.tree.prefix_ids.get(prefix)
        nodes = [nid] if nid is not None else []
        if nid is None or int(self.tree.tree[nid]['source_node']) != int(self.node_id):
            # Preserve only screened suffixes of the actually reattached node.
            nodes = self.tree.by_source.get(int(self.node_id), [])
            self.choice['prefix_recovery'].append({'day': len(prefix), 'node': int(self.node_id), 'prefixes': len(nodes)})
        available = {(int(e['src']), int(e['dst']), str(e['example'])): e for e in edges}
        allowed = {}
        for node in nodes:
            for opt in self.tree.rules.get(node, []):
                key = int(opt['src']), int(opt['dst']), str(opt['example'])
                if key in available:
                    allowed.setdefault(key, (available[key], opt))
        candidates = list(allowed.values())
        if not candidates:
            self.choice.setdefault('node_errors', []).append({'day': len(prefix), 'reason': 'no_legal_screened_suffix'})
            return None
        if len(candidates) == 1:
            chosen = candidates[0][0]
        else:
            self.market_seller.update(obs)
            version = getattr(self.model, 'feature_version', None)
            if version == PRODUCTION_MARKET_VERSION:
                ctx = observation_context(obs)
                scores = [0.] * len(candidates)
                encoder = production_market_option_features
            else:
                ctx = context(obs, self.market_seller)
                scores = [observable_option_score(o, ctx) for _, o in candidates]
                encoder = compact_option_features if version == FEATURE_VERSION else option_features
            vectors = [encoder(self, obs, o, ctx, s) for (_, o), s in zip(candidates, scores)]
            with torch.no_grad():
                probs = torch.softmax(self.model(torch.tensor(vectors, dtype=torch.float32),
                                                torch.tensor(scores, dtype=torch.float32)), dim=0).tolist()
            if self.sample:
                selected = self.actor_rng.choices(range(len(probs)), weights=probs, k=1)[0]
            else:
                selected = max(range(len(probs)), key=probs.__getitem__)
            chosen = candidates[selected][0]
            self.decisions.append({'features': vectors, 'observable_scores': scores, 'action': selected,
                                   'old_logp': math.log(max(probs[selected], 1e-12)), 'old_probs': probs,
                                   'day': len(prefix), 'dst': int(chosen['dst'])})
            # Financing is a soft, public-information input, never a hard ban.
            # A day's total bill is NOT payable all at dawn. No paths are deleted.
            self.choice['node_risk'].append({'day': len(prefix), 'options': len(candidates),
                'selected': selected, 'shortfalls': [affordability_vector(o['next_day_contract'], ctx)[1]
                                                     for _, o in candidates], 'hard_blocked': 0})
        if not self._opened:
            self._opened = True
            self.root_id = int(chosen['dst'])
            self.choice['root'] = self.root_id
        return chosen

    def resolved_route(self):
        prefix = tuple((int(r['dst']), str(r['loc'])) for r in self.choice['path'] if r.get('dst') is not None)
        node = self.tree.prefix_ids.get(prefix)
        if node is not None and 'terminal_route' in self.tree.tree[node]:
            return self.tree.routes[int(self.tree.tree[node]['terminal_route'])]
        return None
