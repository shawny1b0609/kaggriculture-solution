"""Opt-in public-belief market competition; existing farm constraints are inherited."""
from __future__ import annotations

from collections import Counter
import copy
from functools import lru_cache
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'featured_forest_20260925'), str(ROOT)]

from runtime.seller import sale_revenue, shop_demand
from runtime.view import own_farm, rival_farm, step_of
from tools.reliable_market_seller import ReliableMarketSeller
from market_game_20260930.ledger import OwnFlowLedger, PRODUCTS, visible_harvest, engine

VERSION = 'public_belief_duel_v1'


def dual_sale(item, ours, theirs, inventory, params=None, price_fn=None):
    """Exact two-sided receipts for one official same-slot SELL pair."""
    ours, theirs = max(0, int(ours)), max(0, int(theirs))
    inv = int(inventory)
    revenues = [0, 0]
    for unit in range(max(ours, theirs)):
        price = price_fn(inv) if price_fn else engine.market_price(item, inv, params)
        active0, active1 = unit < ours, unit < theirs
        revenues[0] += price * active0
        revenues[1] += price * active1
        if price > 1:
            inv += int(active0) + int(active1)
    return revenues[0], revenues[1], inv


class LedgerSeller(ReliableMarketSeller):
    def __init__(self):
        super().__init__()
        self.ledger = OwnFlowLedger()
        self.rival_backlog = {p: 0.0 for p in PRODUCTS}
        self.rival_net = {p: 0 for p in PRODUCTS}
        self.rival_previous_farm = None
        self.audit = Counter()

    def record_action(self, obs, action, configuration=None):
        self.ledger.record(obs, action, configuration)

    def update(self, obs):
        step = step_of(obs)
        if self.last_step == step:
            return
        if self.last_step is not None and step < self.last_step:
            self.ledger = OwnFlowLedger()
            self.opponent_pressure = {}
            self.rival_backlog = {p: 0.0 for p in PRODUCTS}
            self.rival_previous_farm = None
        net = self.ledger.observe(obs)
        other = rival_farm(obs)
        harvested = visible_harvest(self.rival_previous_farm, other) if self.rival_previous_farm is not None else {}
        for p in PRODUCTS:
            old = float(self.opponent_pressure.get(p) or 0)
            if p in net:
                self.opponent_pressure[p] = .75 * old + .25 * max(0, net[p])
                self.rival_net[p] = net[p]
            else:
                self.opponent_pressure[p] = .75 * old
            # A belief about unsold harvested goods, NOT the opponent's shed.
            # Public disappearances can include losses; decay and scenarios
            # explicitly prevent treating this quantity as a known resource.
            backlog = .985 * self.rival_backlog[p] + float(harvested.get(p, 0))
            backlog -= max(0, net.get(p, 0))
            self.rival_backlog[p] = max(0., min(100., backlog))
        total = sum(self.rival_backlog.values())
        if total > 100:
            self.rival_backlog = {p: q * 100 / total for p, q in self.rival_backlog.items()}
        self.rival_previous_farm = copy.deepcopy(other)
        self.last_inventory = dict(obs.get('market', {}).get('inventory') or {})
        self.last_step = step
        self.own_last_sales = {}


class CompetitiveSeller(LedgerSeller):
    """Finite-action two-stage best response with a conservative rival mixture.

    The horizon liquidation is only a proxy. Entire official games decide if
    it helps terminal margin; no claim of equilibrium or terminal optimality.
    Opponent current-action priors stay fixed across our candidate actions.
    """
    def __init__(self, window=24, rival_weight=1.0):
        super().__init__()
        self.window = int(window)
        self.rival_weight = float(rival_weight)
        self.quantity_fractions = (0., .25, .5, .75, 1.)
        self.duel_details = {}

    def _timed_quantity(self, obs, item, stock, projected_arrival, visible_supply):
        original = super()._timed_quantity(obs, item, stock, projected_arrival, visible_supply)
        if float(own_farm(obs).get('money') or 0) < 10000 or stock <= 0 or step_of(obs) >= 718:
            return original
        step = step_of(obs)
        horizon = min(self.window, 718 - step)
        market = obs.get('market') or {}
        params = engine._resolve_market_params(market.get('params'))
        inventory = int((market.get('inventory') or {}).get(item, params[item]['I0']))
        shops = (obs.get('town') or {}).get('unlocked_shops') or []
        demand = sum(shop_demand(shops, item, t) for t in range(step, step + horizon))
        pressure = max(0., float(self.opponent_pressure.get(item) or 0))
        backlog = max(self.rival_backlog[item], min(40., 4 * pressure))
        future_supply = max(0., float(visible_supply.get(item) or 0)) * min(1., horizon / self.horizon_turns)
        if backlog + future_supply < .5:
            self.audit['no_rival_overlap_fallback'] += 1
            return original
        own_arrivals = int(max(0, projected_arrival) * min(1., horizon / max(1, 24 - int(obs.get('hour') or 0))))
        candidates = sorted({original, stock, 0, *[math.ceil(stock * f) for f in self.quantity_fractions]})
        results = {q: [] for q in candidates}
        own_results = {q: [] for q in candidates}
        rival_results = {q: [] for q in candidates}
        price = lru_cache(maxsize=None)(lambda inv: engine.market_price(item, inv, params))

        @lru_cache(maxsize=None)
        def sell(q, r, inv):
            return dual_sale(item, q, r, inv, params, price)

        # Every own candidate sees identical hidden-stock hypotheses.
        worlds = [(int(round(backlog * scale)), int(round(future_supply * scale)))
                  for scale in (.5, 1., 1.5)]
        weights = (.25, .5, .25)
        cash_other = float(rival_farm(obs).get('money') or 0)
        behavior_weights = (.65, .05, .30) if cash_other < 10000 else (.35, .15, .50)
        for current, later in worlds:
            current, later = min(100, current), min(100, later)
            other_candidates = sorted({0, current, *[math.ceil(current * f) for f in self.quantity_fractions]})
            matrix = {}
            for q in candidates:
                for r in other_candidates:
                    now0, now1, cursor = sell(q, r, inventory)
                    future0, future1, _ = sell(stock - q + own_arrivals, current - r + later, cursor - demand)
                    matrix[q, r] = (now0 + future0, now1 + future1)
            # A revenue-oriented rival responds to a fixed reference action,
            # not secretly to each of our simultaneous proposed instructions.
            best_response = max(other_candidates, key=lambda r: (matrix[original, r][1], r))
            styles = (current, 0, best_response)
            for q in candidates:
                own_values = [matrix[q, r][0] for r in styles]
                rival_values = [matrix[q, r][1] for r in styles]
                expected = sum(w * (a - self.rival_weight * b)
                               for w, a, b in zip(behavior_weights, own_values, rival_values))
                worst = min(a - self.rival_weight * b
                            for r in other_candidates for a, b in [matrix[q, r]])
                results[q].append(.8 * expected + .2 * worst)
                own_results[q].append(sum(w * v for w, v in zip(behavior_weights, own_values)))
                rival_results[q].append(sum(w * v for w, v in zip(behavior_weights, rival_values)))
        scores = {}
        for q in candidates:
            mean = sum(w * v for w, v in zip(weights, results[q]))
            downside = math.sqrt(sum(w * max(0., mean - v) ** 2 for w, v in zip(weights, results[q])))
            carry = .002 * horizon * (stock - q) * params[item]['base']
            scores[q] = mean - .2 * downside - carry
        chosen = max(candidates, key=lambda q: (scores[q], q))
        chosen = max(chosen, min(stock, self.overdue.get(item, 0)))
        self.audit['duel_calls'] += 1
        self.audit['quantity_changed'] += int(chosen != original)
        self.audit['extra_units_requested'] += max(0, chosen - original)
        self.audit['units_deferred'] += max(0, original - chosen)
        detail = dict(reason='PUBLIC_BELIEF_CASH_MARGIN', horizon=horizon,
                      quantity=chosen, reference_quantity=original,
                      rival_weight=self.rival_weight, backlog_belief=backlog,
                      known_demand=demand, scores=scores, worlds=worlds,
                      own_expected={q:sum(w*v for w,v in zip(weights, own_results[q])) for q in candidates},
                      rival_expected={q:sum(w*v for w,v in zip(weights, rival_results[q])) for q in candidates})
        self.duel_details[item] = detail
        self.timing_details[item] = detail
        return chosen

    def plan(self, obs, available, reserved, **kwargs):
        self.duel_details = {}
        plan = super().plan(obs, available, reserved, **kwargs)
        # Do not reorder funding, room-release or overdue obligations.
        if (len(plan.orders) > 1 and not plan.forced
                and float(own_farm(obs).get('money') or 0) >= 10000
                and not any(self.overdue.get(o[1], 0) for o in plan.orders)):
            params = engine._resolve_market_params(obs['market'].get('params'))
            invs = obs['market']['inventory']
            def urgency(order):
                _, p, q = order
                r = int(round(max(self.rival_backlog[p], min(40, 4 * self.opponent_pressure.get(p, 0)))))
                our_first, inv = sale_revenue(p, q, invs[p], params)
                their_after, _ = sale_revenue(p, r, inv, params)
                their_first, inv = sale_revenue(p, r, invs[p], params)
                our_after, _ = sale_revenue(p, q, inv, params)
                return ((our_first - self.rival_weight * their_after)
                        - (our_after - self.rival_weight * their_first), our_first)
            reordered = sorted(plan.orders, key=urgency, reverse=True)
            self.audit['order_reordered'] += int(reordered != plan.orders)
            plan.orders = reordered
        return plan
