"""Observable branch features with optional cash and financing coordinates."""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import SELLABLE
from node_rl.affordability import affordability_vector, next_day_contract
from runtime.seller import shop_demand
from runtime.view import counts, rival_farm


def feature_names(include_cash: bool = False, include_affordability: bool = False) -> list[str]:
    if include_affordability and not include_cash:
        raise ValueError("affordability protocol includes the observed cash coordinate")
    names = [f"fraction_{item}" for item in SELLABLE]
    names += [f"avg_daily_{item}" for item in SELLABLE]
    names += ["segment_days"]
    for group in ("fraction_x_demand", "avg_x_demand", "avg_x_price",
                  "avg_x_rival_farm", "avg_x_rival_sales", "avg_x_cash_gap"):
        names += [f"{group}_{item}" for item in SELLABLE]
    if include_cash:
        names.append("own_cash_per_100k_capped_at_2")
    if include_affordability:
        names.extend(("next_day_required_payment_per_10k",
                      "next_day_cash_shortfall_per_10k",
                      "next_day_rough_order_pressure_per_10"))
    return names


def segment_features(tree: list[dict], routes: dict[int, dict], tapes=None) -> dict[int, dict]:
    specs = {}
    for parent in tree:
        if len(parent["children"]) < 2:
            continue
        for child_id in parent["children"]:
            child = tree[child_id]
            cursor = child
            while len(cursor["children"]) == 1:
                cursor = tree[cursor["children"][0]]
            start = int(parent["day"])
            stop = int(cursor["day"]) if cursor["children"] else 30
            days = max(1, stop - start)
            route_ids = child["routes"]
            totals = {item: 0.0 for item in SELLABLE}
            for route_id in route_ids:
                daily = routes[int(route_id)]["daily_production"]
                for day in range(start, min(stop, len(daily))):
                    for item in SELLABLE:
                        totals[item] += float(daily[day].get(item) or 0)
            avg = {item: totals[item] / max(1, len(route_ids)) / days for item in SELLABLE}
            mass = sum(avg.values())
            specs[child_id] = {
                "start_day": start, "end_day_exclusive": stop, "days": days,
                "avg_daily": avg,
                "fractions": {item: avg[item] / mass if mass else 0.0 for item in SELLABLE},
                "route_count": len(route_ids),
            }
            if tapes is not None:
                edge = routes[min(route_ids)]["edges"][start]
                specs[child_id]["next_day_contract"] = next_day_contract(
                    tapes.day_tape(str(edge["example"]), start))
    return specs


def context(obs: dict, seller) -> dict:
    market = obs.get("market") or {}
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    step = int(obs.get("step") or ((obs.get("day") or 0) * 24 + (obs.get("hour") or 0)))
    own = (obs.get("farms") or [{}, {}])[int(obs.get("player") or 0)]
    rival = counts(rival_farm(obs))
    product_of = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
    rival_prod = {item: float((rival.get("crops") or {}).get(item) or 0) for item in SELLABLE}
    for animal, item in product_of.items():
        rival_prod[item] += float((rival.get("animals") or {}).get(animal) or 0)
    params = market.get("params") or engine.MARKET_PARAMS
    return {
        "cash": float(own.get("money") or 0), "day": int(obs.get("day") or 0),
        "shops": shops,
        "seeds": dict((obs.get("private") or {}).get("seeds") or {}),
        "shed": dict((obs.get("private") or {}).get("shed") or {}),
        "market_prices": dict(market.get("prices") or {}),
        "demand": {item: sum(shop_demand(shops, item, tick)
                            for tick in range(step, min(step + 24, 719)))
                   for item in SELLABLE},
        "price_ratio": {item: float((market.get("prices") or {}).get(item) or 0)
                        / max(1.0, float(params[item]["base"])) for item in SELLABLE},
        "rival_farm": rival_prod,
        "rival_sales_pressure": {item: float(seller.opponent_pressure.get(item) or 0)
                                for item in SELLABLE},
    }


def option_vector(spec: dict, ctx: dict, *, include_cash: bool = False,
                  include_affordability: bool = False) -> list[float]:
    if include_affordability and not include_cash:
        raise ValueError("affordability protocol includes the observed cash coordinate")
    fraction = [spec["fractions"][item] for item in SELLABLE]
    avg = [min(100.0, spec["avg_daily"][item]) / 100.0 for item in SELLABLE]
    demand = [min(24.0, ctx["demand"][item]) / 24.0 for item in SELLABLE]
    price = [min(3.0, max(0.0, ctx["price_ratio"][item])) / 3.0 for item in SELLABLE]
    rivals = [min(30.0, ctx["rival_farm"][item]) / 30.0 for item in SELLABLE]
    sales = [min(5.0, ctx["rival_sales_pressure"][item]) / 5.0 for item in SELLABLE]
    cash_need = max(0.0, min(1.0, (10000.0 - ctx["cash"]) / 10000.0))
    values = (
        fraction + avg + [min(28, spec["days"]) / 28.0]
        + [fraction[i] * demand[i] for i in range(len(SELLABLE))]
        + [avg[i] * demand[i] for i in range(len(SELLABLE))]
        + [avg[i] * price[i] for i in range(len(SELLABLE))]
        + [avg[i] * rivals[i] for i in range(len(SELLABLE))]
        + [avg[i] * sales[i] for i in range(len(SELLABLE))]
        + [avg[i] * cash_need for i in range(len(SELLABLE))]
    )
    if include_cash:
        values.append(min(2.0, max(0.0, ctx["cash"]) / 100000.0))
    if include_affordability:
        values.extend(affordability_vector(spec["next_day_contract"], ctx))
    return values
