"""PM16.1: segment production plus current public market conditions.

Nine product quantities encode BOTH composition and production scale. The
remaining seven coordinates describe time, own cash, one preparation-cost
estimate and three production-weighted market signals. No opponent fields,
route identifiers, distant descendant production or future market observations.
The legacy C20 and 213-coordinate encoders remain available for old policies.
"""
from __future__ import annotations

import math
from torch import nn
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import SELLABLE
from runtime.seller import shop_demand
from runtime.view import own_farm, step_of

FEATURE_VERSION = 'PM16.1'
FEATURE_NAMES = [f'daily_{product.lower()}' for product in SELLABLE] + [
    'step', 'segment_days', 'own_cash', 'next_day_input_pressure',
    'weighted_demand_coverage', 'weighted_price_premium', 'weighted_price_momentum',
]


def observation_context(obs):
    """Only own resources and already visible shops/prices; no rival context."""
    step = step_of(obs)
    private, market = obs.get('private') or {}, obs.get('market') or {}
    shops = list((obs.get('town') or {}).get('unlocked_shops') or [])
    return {
        'cash': max(0., float(own_farm(obs).get('money') or 0)),
        'day': int(obs.get('day') or 0),
        'seeds': dict(private.get('seeds') or {}),
        'shed': dict(private.get('shed') or {}),
        'market_prices': dict(market.get('prices') or {}),
        # Same public 24-hour demand window as the existing selector. Unknown
        # future shop openings are deliberately not used.
        'demand': {p: sum(shop_demand(shops, p, tick)
                          for tick in range(step, min(step + 24, 719)))
                   for p in SELLABLE},
    }


def next_day_input_cost(option, ctx, params):
    """Approximate next-day purchases/hires, not an all-at-dawn cash constraint."""
    contract = option['next_day_contract']
    bill = float(contract['hire_cost']) + float(contract['land_cost'])
    for product, quantity in contract['seeds'].items():
        bill += max(0., float(quantity) - float(ctx['seeds'].get(product) or 0)) * engine.CROPS[product]['seed']
    for product, quantity in contract['shed'].items():
        unit = (engine.ANIMALS[product]['cost'] if product in engine.ANIMALS else
                float(ctx['market_prices'].get(product) or params[product]['base']))
        bill += max(0., float(quantity) - float(ctx['shed'].get(product) or 0)) * unit
    return max(0., bill)


def option_features(agent, obs, option, ctx=None, value_score=0.):
    # value_score is accepted only for interface compatibility, never an input.
    if ctx is None:
        ctx = observation_context(obs)
    params = engine._resolve_market_params((obs.get('market') or {}).get('params'))
    daily = {p: max(0., float(option['avg_daily'].get(p) or 0)) for p in SELLABLE}
    values = [daily[p] / (25. + daily[p]) for p in SELLABLE]
    value = {p: daily[p] * float(params[p]['base']) for p in SELLABLE}
    mass = sum(value.values())
    demand_coverage = premium = momentum = 0.
    for product in SELLABLE:
        weight = value[product] / mass if mass else 0.
        spot = max(1., float(ctx['market_prices'].get(product) or params[product]['base']))
        ema = max(1., float(getattr(agent, 'price_ema', {}).get(product) or spot))
        demand = max(0., float(ctx['demand'].get(product) or 0))
        demand_coverage += weight * math.tanh(math.log((1 + demand) / (1 + daily[product])))
        premium += weight * math.tanh(math.log(spot / max(1., params[product]['base'])))
        momentum += weight * math.tanh(4 * math.log(spot / ema))
    cash = max(0., float(ctx['cash']))
    bill = next_day_input_cost(option, ctx, params)
    values += [
        min(1., max(0., step_of(obs)) / 719),
        min(1., max(0., float(option['days'])) / 30),
        min(1., math.log1p(cash) / math.log1p(300000)),
        bill / (500. + cash + bill),
        demand_coverage, premium, momentum,
    ]
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) and -1 <= v <= 1 for v in values):
        raise ValueError('production/market feature protocol violated')
    return values


class OptionNet(nn.Module):
    feature_version = FEATURE_VERSION

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.Tanh(), nn.Linear(64, 32),
                                 nn.Tanh(), nn.Linear(32, 1))
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, observable_scores=None):
        # No hidden rule score; actor and critic both receive exactly PM16.
        return self.net(features).squeeze(-1)
