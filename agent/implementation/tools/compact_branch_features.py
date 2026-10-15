"""Twenty public, tree-independent coordinates per candidate (protocol C20.1).

Products are pooled using the candidate's production VALUE shares, not raw
counts (one melon is not one wheat). No route identity or future market data.
The legacy 213-coordinate encoder remains untouched for old checkpoints.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import SELLABLE
from runtime.view import own_farm, rival_farm

FEATURE_VERSION = 'C20.1'
FEATURE_NAMES = [
    'day', 'own_cash', 'cash_difference', 'execution_debt',
    'segment_days', 'next_choice_flexibility', 'segment_daily_base_value',
    'continuation_daily_base_value', 'production_diversity', 'animal_value_share',
    'weighted_price_premium', 'weighted_price_momentum', 'weighted_inventory_pressure',
    'weighted_demand_coverage', 'weighted_rival_supply_pressure',
    'next_day_bill_to_liquidity', 'next_day_shortfall', 'hire_land_to_cash',
    'next_day_peak_farm_orders', 'shed_fill',
]


def option_features(agent, obs, option, ctx, value_score=0.):
    # value_score is a legacy call-signature argument, NOT another input.
    cash = max(0., float(own_farm(obs).get('money') or 0))
    other_cash = max(0., float(rival_farm(obs).get('money') or 0))
    market = obs.get('market') or {}
    params = engine._resolve_market_params(market.get('params'))
    daily = {p: max(0., float(option['avg_daily'].get(p) or 0)) for p in SELLABLE}
    remaining = option.get('remaining_daily') or daily
    base_value = sum(daily[p] * params[p]['base'] for p in SELLABLE)
    # A setup-only segment still needs to describe its reachable industries.
    # This continuation mix is potential, not a promised simultaneous output.
    profile = daily if base_value > 0 else remaining
    value = {p: max(0., float(profile.get(p) or 0)) * params[p]['base'] for p in SELLABLE}
    mass = sum(value.values())
    weights = {p: value[p] / mass if mass else 0. for p in SELLABLE}
    continuation = sum(max(0., float(remaining.get(p) or 0)) * params[p]['base'] for p in SELLABLE)
    premium = momentum = inventory_pressure = demand_coverage = rival_pressure = 0.
    for p in SELLABLE:
        weight, rule = weights[p], params[p]
        spot = max(1., float((market.get('prices') or {}).get(p) or rule['base']))
        ema = max(1., float(getattr(agent, 'price_ema', {}).get(p) or spot))
        inv = float((market.get('inventory') or {}).get(p, rule['I0']))
        output = max(0., float(profile.get(p) or 0))
        demand = max(0., float(ctx['demand'].get(p) or 0))
        rival_sales = 24 * max(0., float(ctx['rival_sales_pressure'].get(p) or 0))
        rival_field = max(0., float(ctx['rival_farm'].get(p) or 0))
        premium += weight * math.tanh(math.log(spot / max(1., rule['base'])))
        momentum += weight * math.tanh(4 * math.log(spot / ema))
        inventory_pressure += weight * math.tanh((inv - rule['I0']) / max(1., rule['T']))
        demand_coverage += weight * math.tanh(math.log((1 + demand) / (1 + output)))
        rival_pressure += weight * math.tanh((rival_sales + rival_field) / (1 + demand + output))
    contract = option['next_day_contract']
    private = obs.get('private') or {}
    shed, seeds = private.get('shed') or {}, private.get('seeds') or {}
    fixed = float(contract['hire_cost']) + float(contract['land_cost'])
    bill = fixed
    for p, qty in contract['seeds'].items():
        bill += max(0., qty - float(seeds.get(p) or 0)) * engine.CROPS[p]['seed']
    for p, qty in contract['shed'].items():
        unit = (engine.ANIMALS[p]['cost'] if p in engine.ANIMALS else
                float(ctx['market_prices'].get(p) or params[p]['base']))
        bill += max(0., qty - float(shed.get(p) or 0)) * unit
    # Only actual free shed stock is liquid; bagged/future harvest is not cash.
    liquid = cash + .7 * sum(max(0., float(shed.get(p) or 0) - float(contract['shed'].get(p) or 0))
                             * float(ctx['market_prices'].get(p) or params[p]['base']) for p in SELLABLE)
    debt = sum(len(getattr(agent, key, [])) for key in
               ('deferred_weed_jobs', 'deferred_cash_jobs', 'market_backlog'))
    values = [
        min(1., max(0., float(obs.get('day') or 0)) / 30),
        min(1., math.log1p(cash) / math.log1p(300000)),
        math.tanh((cash - other_cash) / 50000), math.tanh(debt / 10),
        min(1., float(option['days']) / 30),
        min(1., math.log1p(max(0., float(option.get('next_choice_options') or 0))) / math.log1p(64)),
        math.tanh(base_value / 5000), math.tanh(continuation / 5000),
        (1 - sum(w * w for w in weights.values())) if mass else 0.,
        sum(weights[p] for p in ('MILK', 'WOOL', 'EGG')),
        premium, momentum, inventory_pressure, demand_coverage, rival_pressure,
        math.tanh(bill / (500 + liquid)),
        math.tanh(max(0., bill - liquid) / (500 + cash)),
        math.tanh(fixed / (500 + cash)),
        min(1., float(option.get('peak_farm_orders') or 0) / 10),
        min(1., sum(max(0., float(v or 0)) for v in shed.values()) / 100),
    ]
    if len(values) != 20 or not all(math.isfinite(v) and -1 <= v <= 1 for v in values):
        raise ValueError('compact feature protocol violated')
    return values


class OptionNet(nn.Module):
    feature_version = FEATURE_VERSION

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 32),
                                 nn.Tanh(), nn.Linear(32, 1))
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, observable_scores=None):
        # No hidden 21st heuristic score: both policy and critic see only C20.
        return self.net(features).squeeze(-1)
