"""Clip market orders to cash, shed, seeds, and shop stock. No empty-handed actions."""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as ANIMAL_RULES,
    CROPS as CROP_RULES,
    LAND_PRICES,
    _hire_cost,
    market_price,
)


def clip_market(orders: list, farm: dict, private: dict, obs: dict | None = None) -> list:
    cash = float(farm.get("money") or 0)
    shed = {k: int(v or 0) for k, v in (private.get("shed") or {}).items()}
    seeds = {k: int(v or 0) for k, v in (private.get("seeds") or {}).items()}
    already = int(farm.get("hires_today") or 0)
    extra = max(0, len(farm.get("unlocked_quadrants") or []) - 1)
    market = (obs or {}).get("market") or {}
    inv = dict(market.get("inventory") or {})
    params = market.get("params")
    out = []
    for raw in orders or []:
        if not raw:
            continue
        op = str(raw[0])
        if op == "HIRE":
            cost = _hire_cost(already)
            if cash < cost:
                continue
            out.append(["HIRE"])
            cash -= cost
            already += 1
            continue
        if op == "BUY_LAND":
            if extra >= len(LAND_PRICES):
                continue
            price = LAND_PRICES[extra]
            if cash < price:
                continue
            out.append(["BUY_LAND"])
            cash -= price
            extra += 1
            continue
        if op == "BUY_SEED" and len(raw) >= 3:
            item = str(raw[1])
            spec = CROP_RULES.get(item)
            if not spec:
                continue
            unit = int(spec["seed"])
            want = int(raw[2] or 0)
            n = min(want, int(cash // unit) if unit else 0)
            if n <= 0:
                continue
            out.append(["BUY_SEED", item, n])
            cash -= n * unit
            seeds[item] = int(seeds.get(item) or 0) + n
            continue
        if op == "BUY_ANIMAL" and len(raw) >= 3:
            item = str(raw[1])
            spec = ANIMAL_RULES.get(item)
            if not spec:
                continue
            unit = int(spec["cost"])
            want = int(raw[2] or 0)
            n = min(want, int(cash // unit) if unit else 0)
            if n <= 0:
                continue
            out.append(["BUY_ANIMAL", item, n])
            cash -= n * unit
            continue
        if op == "BUY_PRODUCT" and len(raw) >= 3:
            item = str(raw[1])
            want = int(raw[2] or 0)
            stock = int(inv.get(item) or 0)
            bought = 0
            for _ in range(max(0, want)):
                if stock <= 0:
                    break
                price = market_price(item, stock, params)
                if cash < price:
                    break
                cash -= price
                stock -= 1
                bought += 1
            if bought <= 0:
                continue
            out.append(["BUY_PRODUCT", item, bought])
            inv[item] = stock
            continue
        if op == "SELL" and len(raw) >= 3:
            item = str(raw[1])
            have = int(shed.get(item) or 0)
            n = min(int(raw[2] or 0), have)
            if n <= 0:
                continue
            out.append(["SELL", item, n])
            shed[item] = have - n
            continue
        out.append(list(raw))
    return out[:10]
