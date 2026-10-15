"""Reverse a source day into minimum dawn stock and labor requirements.

Unit deltas come from the source's effective actions. Market purchases occur
after units each hour. The prefix minimum therefore catches a PLANT/PICKUP at
hour 0 that cannot be supported by an hour-0 purchase. This is a stock/timing
contract, not a proof of cash affordability, capacity, or future market fills.
"""
from __future__ import annotations

from collections import defaultdict


def advance_purchase_horizon(tape: list[dict], private: dict | None = None,
                             order_limit: int = 10) -> list[dict]:
    """Buy ahead when the next market has too few slots for its next inputs.

    `tape[0]` is the next field phase. Its market can normally provision
    `tape[1]`, except when staffing/land leave fewer slots than the number of
    distinct required item orders. Estimate this after the source worker deltas
    and the stock purchased for the current horizon. This is a slot contract,
    not a cash/warehouse-capacity proof.
    """
    if not tape:
        return []
    initial = {
        "seeds": {k: int(v or 0) for k, v in ((private or {}).get("seeds") or {}).items()},
        "shed": {k: int(v or 0) for k, v in ((private or {}).get("shed") or {}).items()},
    }
    count = 1
    while count < len(tape):
        contract = dawn_requirements(tape[:count], rely_on_intraday_buys=False)
        held = {bucket: dict(initial[bucket]) for bucket in ("seeds", "shed")}
        for bucket in ("seeds", "shed"):
            for item, amount in contract[bucket].items():
                held[bucket][item] = max(int(held[bucket].get(item) or 0), int(amount))
        for step in tape[:count]:
            events = step.get("unit_events")
            if events is None:
                events = [{"seeds": step.get("seed_delta_units") or {},
                           "shed": step.get("shed_delta_units") or {}}]
            for event in events:
                for bucket in ("seeds", "shed"):
                    for item, delta in (event.get(bucket) or {}).items():
                        held[bucket][item] = int(held[bucket].get(item) or 0) + int(delta)
        prior = tape[count - 1]
        positions = prior.get("positions") or []
        hands_now = max(0, len(positions) - 1)
        post_hands = prior.get("post_hands")
        hires = max(0, int(post_hands) - hands_now) if post_hands is not None else 0
        land_now = int(prior.get("land") or 1)
        post_land = prior.get("post_land")
        land = max(0, int(post_land) - land_now) if post_land is not None else 0
        next_need = dawn_requirements(tape[count:count + 1], rely_on_intraday_buys=False)
        item_orders = sum(
            int(int(amount) > int(held[bucket].get(item) or 0))
            for bucket in ("seeds", "shed")
            for item, amount in next_need[bucket].items()
        )
        if item_orders <= max(0, order_limit - hires - land):
            break
        count += 1
    return tape[:count]


def dawn_requirements(tape: list[dict], rely_on_intraday_buys: bool = False,
                      include_source_sales: bool = False) -> dict:
    seeds = defaultdict(int)
    shed = defaultdict(int)
    min_seeds = defaultdict(int)
    min_shed = defaultdict(int)
    hires_needed_by_hour = {}
    for hour, step in enumerate(tape):
        events = step.get("unit_events")
        if events is None:
            events = [{"seeds": step.get("seed_delta_units") or {}, "shed": step.get("shed_delta_units") or {}}]
        for event in events:
            for item, delta in (event.get("seeds") or {}).items():
                seeds[item] += int(delta)
                min_seeds[item] = max(min_seeds[item], -seeds[item])
            for item, delta in (event.get("shed") or {}).items():
                shed[item] += int(delta)
                min_shed[item] = max(min_shed[item], -shed[item])
        units = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
        active_hand = max((i for i, action in enumerate(units) if i > 0 and isinstance(action, list) and action and action[0] != "PASS"), default=0)
        if active_hand:
            hires_needed_by_hour[hour] = active_hand
        for order in step.get("market") or []:
            if not isinstance(order, list) or not order or len(order) < 3:
                continue
            op, item, qty = str(order[0]), str(order[1]), int(order[2] or 0)
            if qty <= 0:
                continue
            if include_source_sales and op == "SELL":
                shed[item] -= qty
                min_shed[item] = max(min_shed[item], -shed[item])
            elif not rely_on_intraday_buys:
                continue
            elif op == "BUY_SEED":
                seeds[item] += qty
            elif op in ("BUY_ANIMAL", "BUY_PRODUCT"):
                shed[item] += qty
            # A future source SELL is normally discretionary for *production*,
            # but it may finance a later seed or hire order. The seller's
            # surplus check can include those source sales as obligations so
            # an earlier extra sale does not silently steal their stock.
    return {
        "seeds": {item: n for item, n in min_seeds.items() if n > 0},
        "shed": {item: n for item, n in min_shed.items() if n > 0},
        "hires_needed_by_hour": hires_needed_by_hour,
        "max_active_hands": max(hires_needed_by_hour.values(), default=0),
        "relies_on_intraday_buys": rely_on_intraday_buys,
        "includes_source_sales": include_source_sales,
    }


def contract_order_gaps(contract: dict, private: dict) -> tuple[list[list], list[dict]]:
    """Translate a stock contract into complete, uncapped purchase gaps.

    Order-count, cash, and shed-capacity limits are deliberately left to the
    caller.  Keeping the complete intent separate from what can be submitted in
    one market phase lets a runtime retain and retry the remainder instead of
    silently dropping it.
    """
    seeds = private.get("seeds") or {}
    shed = private.get("shed") or {}
    requested = []
    unmet = []
    for item, minimum in sorted((contract.get("seeds") or {}).items()):
        gap = int(minimum) - int(seeds.get(item) or 0)
        if gap > 0:
            requested.append(["BUY_SEED", item, gap])
    for item, minimum in sorted((contract.get("shed") or {}).items()):
        gap = int(minimum) - int(shed.get(item) or 0)
        if gap <= 0:
            continue
        op = "BUY_ANIMAL" if item in ("COW", "SHEEP", "GOOSE") else "BUY_PRODUCT"
        if op == "BUY_PRODUCT" and item not in ("WHEAT", "FERTILIZER"):
            unmet.append({"op": op, "item": item, "quantity": gap, "reason": "not_buyable"})
            continue
        requested.append([op, item, gap])
    return requested, unmet


def first_stock_use_offsets(tape: list[dict]) -> dict[tuple[str, str], int]:
    """Return the first future field offset that consumes each stock item."""
    result: dict[tuple[str, str], int] = {}
    for hour, step in enumerate(tape):
        events = step.get("unit_events")
        if events is None:
            events = [{"seeds": step.get("seed_delta_units") or {},
                       "shed": step.get("shed_delta_units") or {}}]
        for event in events:
            for item, delta in (event.get("seeds") or {}).items():
                if int(delta) < 0:
                    result.setdefault(("BUY_SEED", str(item)), hour)
            for item, delta in (event.get("shed") or {}).items():
                if int(delta) >= 0:
                    continue
                op = "BUY_ANIMAL" if item in ("COW", "SHEEP", "GOOSE") else "BUY_PRODUCT"
                result.setdefault((op, str(item)), hour)
    return result


def top_up_orders(contract: dict, private: dict, max_orders: int = 10, shed_capacity: int = 100) -> tuple[list[list], list[dict]]:
    """Buy only the dawn-prefix deficit, reporting rather than hiding limits."""
    shed = private.get("shed") or {}
    room = max(0, shed_capacity - sum(int(v or 0) for v in shed.values()))
    requested, unmet = contract_order_gaps(contract, private)
    orders = []
    for op, item, qty in requested:
        if len(orders) >= max_orders:
            unmet.append({"op": op, "item": item, "quantity": qty, "reason": "market_order_limit"})
            continue
        if op in ("BUY_ANIMAL", "BUY_PRODUCT"):
            take = min(qty, room)
            if take > 0:
                orders.append([op, item, take])
                room -= take
            if take < qty:
                unmet.append({"op": op, "item": item, "quantity": qty - take, "reason": "shed_capacity"})
        else:
            orders.append([op, item, qty])
    return orders, unmet
