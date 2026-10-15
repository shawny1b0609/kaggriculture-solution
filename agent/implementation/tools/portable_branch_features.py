"""Portable public-information branch features, detached from the old tree package.

Copied mechanically from train_current_tree_rl's feature protocol. The only
model change bounds the heuristic prior to +/-0.25 logits, preserving exploration.
No episode identity, node identity, source cash or future market data is input.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from config import SELLABLE
from node_rl.affordability import affordability_vector
from runtime.view import bag_total, counts, own_farm, rival_farm
PRODUCTS = tuple(SELLABLE)
ANIMAL_PRODUCT = {"MILK": "COW", "WOOL": "SHEEP", "EGG": "GOOSE"}
BASE_TEMPERATURE = 600.0

def feature_names() -> list[str]:
    global_names = [
        "day", "hour", "own_cash", "rival_cash", "cash_difference",
        "own_workers", "rival_workers", "own_land", "rival_land",
        "own_weeds", "rival_weeds", "own_ready_crop", "rival_ready_crop",
        "own_ready_animal", "rival_ready_animal", "own_hungry",
        "rival_hungry", "own_unwatered", "rival_unwatered", "shop_count",
    ]
    per_product = [
        "option_daily", "option_share", "demand_24h", "spot_to_base",
        "market_inventory", "price_vs_ema", "market_flow_ema", "rival_field",
        "own_field", "rival_sale_pressure", "own_sale_ema", "own_shed",
        "own_bags", "own_seeds", "next_day_seed_need", "next_day_shed_need",
        "daily_x_price", "daily_x_demand", "daily_x_rival",
    ]
    option_names = [
        "option_days", "hire_count",
        "land_count", "hire_cost", "land_cost", "next_day_payment",
        "next_day_shortfall", "next_day_order_pressure", "observable_value_score",
    ]
    return (global_names + [f"{label}_{item}" for item in PRODUCTS for label in per_product]
            + option_names + [f"remaining_daily_{item}" for item in PRODUCTS]
            + ['open_weed_jobs', 'open_cash_jobs', 'backlog_orders', 'shed_fill'])


FEATURE_NAMES = feature_names()


def _field_product(farm_counts: dict, item: str) -> float:
    crop = float((farm_counts.get("crops") or {}).get(item) or 0)
    animal = ANIMAL_PRODUCT.get(item)
    return crop + (float((farm_counts.get("animals") or {}).get(animal) or 0) if animal else 0.0)


def _cap(value: float, limit: float) -> float:
    return max(-1.0, min(1.0, float(value) / limit))


def observable_option_score(option: dict, ctx: dict) -> float:
    """Tree-independent, observable production/cash-flow starting score."""
    daily_value = 0.0
    for item in PRODUCTS:
        output = max(0.0, float(option["avg_daily"].get(item) or 0))
        if not output:
            continue
        base = max(1.0, float(option["base_prices"][item]))
        spot = float(ctx["market_prices"].get(item) or base)
        demand = min(1.0, float(ctx["demand"].get(item) or 0) / 12.0)
        rival = float(ctx["rival_farm"].get(item) or 0)
        pressure = float(ctx["rival_sales_pressure"].get(item) or 0)
        daily_value += output * (0.55 * base + 0.45 * spot) * (
            1.0 + 0.15 * demand) / (1.0 + 0.012 * rival + 0.04 * pressure)
    _payment, shortfall, order_pressure = affordability_vector(option["next_day_contract"], ctx)
    cash = max(0.0, float(ctx["cash"]))
    financing = (0.8 if cash < 10000 else 0.35) * 10000 * shortfall / max(1.0, float(option["days"]))
    congested_orders = max(0.0, 10.0 * order_pressure - 8.0) * 120.0
    return daily_value - financing - congested_orders


def option_features(agent, obs: dict, option: dict, ctx: dict, value_score: float) -> list[float]:
    ours = counts(own_farm(obs))
    rival = counts(rival_farm(obs))
    private = obs.get("private") or {}
    bags = bag_total(private)
    prices = (obs.get("market") or {}).get("prices") or {}
    inventory = (obs.get("market") or {}).get("inventory") or {}
    contract = option["next_day_contract"]
    cash = float(ours["cash"])
    rival_cash = float(rival["cash"])
    values = [
        float(obs.get("day") or 0) / 30, float(obs.get("hour") or 0) / 24,
        math.log1p(max(0.0, cash)) / 12, math.log1p(max(0.0, rival_cash)) / 12,
        _cap(cash - rival_cash, 100000),
        ours["workers"] / 16, rival["workers"] / 16,
        ours["land"] / 4, rival["land"] / 4,
        ours["weeds"] / 20, rival["weeds"] / 20,
        ours["ready_crop"] / 20, rival["ready_crop"] / 20,
        ours["ready_animal"] / 20, rival["ready_animal"] / 20,
        ours["hungry"] / 20, rival["hungry"] / 20,
        ours["unwatered"] / 20, rival["unwatered"] / 20,
        len(ctx["shops"]) / 10,
    ]
    daily = option["avg_daily"]
    total_daily = max(1.0, sum(float(daily.get(item) or 0) for item in PRODUCTS))
    for item in PRODUCTS:
        output = float(daily.get(item) or 0)
        base = max(1.0, float(option["base_prices"][item]))
        spot = float(prices.get(item) or base)
        ema = max(1.0, float(agent.price_ema.get(item) or spot))
        demand = float(ctx["demand"].get(item) or 0)
        rival_field = _field_product(rival, item)
        own_field = _field_product(ours, item)
        values.extend([
            min(2.0, output / 25), output / total_daily,
            min(2.0, demand / 24), min(3.0, spot / base) / 2,
            _cap(float(inventory.get(item) or 0), 300),
            _cap(math.log(max(0.01, spot / ema)), 1),
            _cap(float(agent.flow_ema.get(item) or 0), 10),
            min(2.0, rival_field / 20), min(2.0, own_field / 20),
            min(2.0, float(ctx["rival_sales_pressure"].get(item) or 0) / 5),
            min(2.0, float(agent.sales_ema.get(item) or 0) / 5),
            min(2.0, float((private.get("shed") or {}).get(item) or 0) / 100),
            min(2.0, float(bags.get(item) or 0) / 100),
            min(2.0, float((private.get("seeds") or {}).get(item) or 0) / 20),
            min(2.0, float(contract["seeds"].get(item) or 0) / 20),
            min(2.0, float(contract["shed"].get(item) or 0) / 20),
            min(3.0, output * spot / 3000),
            min(3.0, output * demand / 300),
            min(3.0, output * rival_field / 300),
        ])
    payment, shortfall, order_pressure = affordability_vector(contract, ctx)
    values.extend([
        min(1.0, float(option["days"]) / 30),
        min(2.0, float(contract["hire_count"]) / 12),
        min(1.0, float(contract["land_count"]) / 4),
        min(2.0, float(contract["hire_cost"]) / 10000),
        min(2.0, float(contract["land_cost"]) / 10000),
        min(2.0, float(payment)), min(2.0, float(shortfall)),
        min(2.0, float(order_pressure)), _cap(value_score, 6000),
    ])
    values.extend(min(3.0, float((option.get('remaining_daily') or {}).get(p) or 0) / 25)
                  for p in PRODUCTS)
    values.extend([min(2., len(getattr(agent, 'deferred_weed_jobs', [])) / 10),
                   min(2., len(getattr(agent, 'deferred_cash_jobs', [])) / 10),
                   min(2., len(getattr(agent, 'market_backlog', {})) / 10),
                   min(1., sum(float(v or 0) for v in (private.get('shed') or {}).values()) / 100)])
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in values):
        raise ValueError(f"feature protocol violated: got={len(values)} expected={len(FEATURE_NAMES)}")
    return values


class OptionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, observable_scores: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1) + 0.25 * torch.tanh(observable_scores / BASE_TEMPERATURE)



