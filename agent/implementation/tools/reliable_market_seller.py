"""Opt-in market repairs; original farm-order reservations remain authoritative."""
from __future__ import annotations
from collections import deque

from config import SELLABLE
from runtime.seller import RollingLiquiditySeller, sale_revenue, engine
from runtime.view import own_farm, step_of

SELLER_VERSION = 'rolling_observed_age_4_48_arrivals_v1'


class ReliableMarketSeller(RollingLiquiditySeller):
    def __init__(self):
        super().__init__(price_window_turns=48, low_cash_wait_penalty=.01)
        self.bootstrap_window_turns = 4
        self.account_for_arrivals = True
        self.stock_lots = {p: deque() for p in SELLABLE}
        self.stock_step = None
        self.overdue = {}

    def _observe_stock(self, obs, available):
        """Age observed stock, never clear age merely because SELL was proposed.

        Net decreases retire oldest lots; increases create new lots. With only
        stock observations simultaneous deposits/consumption can be ambiguous:
        this deliberately overestimates age, rather than hiding old inventory.
        Production reservations are still subtracted by the original planner.
        """
        step = step_of(obs)
        if self.stock_step is not None and step < self.stock_step:
            self.stock_lots = {p: deque() for p in SELLABLE}
        self.stock_step = step
        limit = 4 if float(own_farm(obs).get('money') or 0) < 10000 else 48
        self.overdue = {}
        for p in SELLABLE:
            lots = self.stock_lots[p]
            have = max(0, int(available.get(p) or 0))
            known = sum(q for _, q in lots)
            if have > known:
                lots.append([step, have - known])
            elif have < known:
                remove = known - have
                while remove and lots:
                    take = min(remove, lots[0][1])
                    lots[0][1] -= take
                    remove -= take
                    if not lots[0][1]:
                        lots.popleft()
            self.overdue[p] = sum(q for born, q in lots if step - born >= limit)

    @staticmethod
    def _visible_supply(farm, day, horizon_days):
        # Zero is a valid planted/placed day, never a missing value.
        supply = {p: 0. for p in SELLABLE}
        for row in farm.get('tiles') or []:
            for tile in row or []:
                if not isinstance(tile, dict):
                    continue
                held = max(0, int(tile.get('yield_units') or 0))
                crop = str(tile.get('crop') or '')
                rule = engine.CROPS.get(crop)
                if tile.get('kind') == 'PLANT' and rule and crop in supply:
                    planted = tile.get('planted_day')
                    age = day - int(day if planted is None else planted)
                    supply[crop] += held
                    if rule.get('ongoing'):
                        for offset in range(1, horizon_days + 1):
                            future = age + offset
                            if future >= rule['first_yield_day'] and (future - rule['first_yield_day']) % max(1, rule['interval']) == 0:
                                supply[crop] += 1
                    elif age < rule['max_yield_day'] <= age + horizon_days:
                        supply[crop] += max(0, rule['max_yield'] - held)
                rule = engine.ANIMALS.get(str(tile.get('animal') or ''))
                if rule and rule['product'] in supply:
                    placed = tile.get('placed_day')
                    age = day - int(day if placed is None else placed)
                    p = rule['product']
                    supply[p] += held
                    for offset in range(1, horizon_days + 1):
                        future = age + offset
                        if future >= rule['first_yield_day'] and (future - rule['first_yield_day']) % max(1, rule['interval']) == 0:
                            supply[p] += 1
        return supply

    def _timed_quantity(self, obs, item, stock, projected_arrival, visible_supply):
        quantity = super()._timed_quantity(obs, item, stock, projected_arrival, visible_supply)
        due = min(stock, self.overdue.get(item, 0))
        if due > quantity:
            self.timing_details[item] = {**self.timing_details.get(item, {}),
                'reason': 'OBSERVED_STOCK_AGE_LIMIT', 'quantity': due,
                'overdue_units': due}
        return max(quantity, due)

    def plan(self, obs, available, reserved, **kwargs):
        self._observe_stock(obs, available)
        plan = super().plan(obs, available, reserved, **kwargs)
        # Keep funding/capacity obligations first; among other signals, overdue
        # inventory must not starve forever behind more expensive products.
        slots = max(0, min(10, int(kwargs.get('available_order_slots', 10))))
        market = obs.get('market') or {}
        inventory, params = market.get('inventory') or {}, market.get('params')
        ranked = []
        for p in SELLABLE:
            quantity = max(plan.timing_quantities.get(p, 0), plan.forced_quantities.get(p, 0))
            quantity = min(quantity, max(0, int(available.get(p) or 0) - int(reserved.get(p) or 0)))
            if quantity:
                revenue, _ = sale_revenue(p, quantity, int(inventory.get(p) or 0), params)
                age = step_of(obs) - self.stock_lots[p][0][0] if self.stock_lots[p] else 0
                ranked.append((int(plan.forced_quantities.get(p, 0) > 0),
                               age if self.overdue.get(p, 0) else -1,
                               revenue / quantity, revenue, p, quantity))
        ranked.sort(reverse=True)
        chosen = ranked[:slots]
        plan.orders = [['SELL', p, q] for _, _, _, _, p, q in chosen]
        plan.expected_revenue = sum(row[3] for row in chosen)
        plan.funding_shortfall = max(0., plan.funding_gap - self.stressed_revenue(obs, plan.orders))
        self.own_last_sales = {p: q for _, p, q in plan.orders}
        return plan
