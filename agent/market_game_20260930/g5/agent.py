"""Rank or synchronize compatible crop cycles; freeze execution/market rules."""
from collections import defaultdict
import copy

from market_game_20260930.g4.agent import (
    CompactRuleAgent, FlexibleCropAgent, TimedFlowSeller, cycle_value,
)
from market_game_20260930.g4.cycles import available_cycles


def delivery_step(cycle, tapes, loc):
    """First market turn able to sell this cargo (field actions precede market)."""
    day, hour, unit = cycle['harvest_day'], cycle['harvest_hour'], cycle['harvest_unit']
    if cycle['delivery'] == 'midnight_auto_drop':
        return (day + 1) * 24
    for h, row in enumerate(tapes.day_tape(loc, day)[hour + 1:], hour + 1):
        commands = [row['farmer'], *row['hands']]
        if unit < len(commands) and commands[unit] == ['DROP']:
            return day * 24 + h
    raise ValueError('certified DROP is missing')


def day_choices(obs, cycles, forecasts, *, selection='rank', recovering=False):
    """Pure public-information ranking. Stable ties preserve earliest planting."""
    if selection not in ('rank', 'first', 'last_forced'):
        raise ValueError(selection)
    safety = ('cash_below_2000' if float(obs['farms'][int(obs['player'])]['money']) < 2000
              else 'unresolved_execution_debt' if recovering else None)
    candidates = cycles[:1] if selection == 'first' else cycles
    rows = []
    for cycle in candidates:
        advantage, details = cycle_value(obs, cycle, forecasts)
        rows.append(dict(id=cycle['id'], day=cycle['plant_day'], xy=cycle['xy'],
                         selected='WHEAT', advantage=advantage, details=details,
                         safety=safety, reason='not_best_today'))
    if not rows:
        return [], rows
    pick = len(rows)-1 if selection == 'last_forced' else max(range(len(rows)), key=lambda i: rows[i]['advantage'])
    selected = selection == 'last_forced' or (safety is None and rows[pick]['advantage'] >= 20)
    rows[pick]['reason'] = ('selected' if selected else safety or 'advantage_below_20')
    if selected:
        rows[pick]['selected'] = 'CARROT'
    return [candidates[pick]] if selected else [], rows


def synchronized_choices(obs, cycles, forecasts, *, recovering=False):
    """Keep the first-cycle decision; add at most one same-arrival crop."""
    chosen, decisions = day_choices(obs, cycles, forecasts, selection='first', recovering=recovering)
    if not chosen:
        return chosen, decisions
    primary = chosen[0]
    decisions[0]['role'] = 'primary'
    eligible = [c for c in cycles if c['id'] != primary['id']
                and c['delivery_step'] == primary['delivery_step']]
    if not eligible:
        return chosen, decisions
    marginal = copy.deepcopy(obs)
    inv = marginal['market']['inventory']
    inv['WHEAT'] -= primary['wheat']['harvested']
    inv['CARROT'] += primary['carrot']['harvested']
    extra, rows = day_choices(marginal, eligible, forecasts, recovering=recovering)
    for row in rows:
        row['role'] = 'same_arrival_extra'
        row['delivery_step'] = primary['delivery_step']
    return chosen + extra, decisions + rows


class RankedCropAgent(FlexibleCropAgent):
    def __init__(self, selection='rank', *, tree=None, cards=None):
        super().__init__(mode='wheat', limit=0, tree=tree, cards=cards)
        if selection not in ('rank', 'first', 'last_forced', 'sync'):
            raise ValueError(selection)
        self.selection = selection
        self.crop_pool = defaultdict(list)
        for cycle in available_cycles():
            if selection == 'sync':
                cycle = copy.deepcopy(cycle)
                cycle['delivery_step'] = delivery_step(cycle, self.tapes, self.tapes.loc)
            self.crop_pool[cycle['plant_day']].append(cycle)

    def _dawn(self, obs, day):
        cycles = self.crop_pool.get(day, [])
        if cycles:
            self.market_seller.update(obs)
            recovering = bool(self.deferred_cash_jobs or self.deferred_weed_jobs or self.pending_weed_jobs)
            if self.selection == 'sync':
                chosen, decisions = synchronized_choices(obs, cycles, self.market_seller.flow_prediction, recovering=recovering)
            else:
                chosen, decisions = day_choices(obs, cycles, self.market_seller.flow_prediction,
                                                selection=self.selection, recovering=recovering)
            for cycle in chosen:
                self.tapes.choose_carrot(cycle)
            self.crop_decisions.extend(decisions)
        # Skip FlexibleCropAgent's fixed-first choice, but reuse the exact same
        # dawn executor, resource compiler, weed recovery and source accounting.
        CompactRuleAgent._dawn(self, obs, day)


def factory(name):
    from market_game_20260930.g4.agent import factory as previous_factory
    from market_game_20260930.players import RecordedMarketAgent
    variants = {'RANK1': 'rank', 'FIRST15': 'first', 'LAST15': 'last_forced', 'SYNC2': 'sync'}
    if name not in variants:
        return previous_factory(name)
    return lambda: RecordedMarketAgent(RankedCropAgent(variants[name]), TimedFlowSeller(True))
