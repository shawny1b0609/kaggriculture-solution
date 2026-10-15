"""Experimental day-end sale to reserve shed room for next-dawn PLACE actions.

At dawn all worker bags fall into the 100-slot shed and overflow is discarded.
The next hour's worker PLACE happens before market sales, so a full shed can
silently shorten a fertilizer/WHEAT deposit and starve a later worker. This
guard changes only discretionary SELL quantities at the preceding h23; it
never changes source field work or required purchase/hire orders.
"""
from __future__ import annotations

from config import SELLABLE
from runtime.grid_agent import _post_unit_private


def next_dawn_place_room(tape: list[dict]) -> int:
    room = 0
    for step in tape[:2]:
        jobs = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
        events = step.get("unit_events") or []
        deposited = 0
        for worker, action in enumerate(jobs):
            if (not isinstance(action, list) or len(action) < 2 or action[0] != "PLACE"
                    or action[1] not in ("WHEAT", "FERTILIZER") or worker >= len(events)):
                continue
            deposited += max(0, int((events[worker].get("shed") or {}).get(action[1]) or 0))
        room = max(room, deposited)
    return min(100, room)


def next_dawn_place_room_ordered(tape: list[dict]) -> int:
    """Peak cumulative shed addition across both field phases, in worker order.

    Consecutive PLACEs accumulate; earlier PICKUPs make space. The legacy
    maximum-of-hourly-deposits misses this dependency and is kept for controls.
    """
    if not next_dawn_place_room(tape):
        return 0
    balance = peak = 0
    for step in tape[:2]:
        for event in step.get('unit_events') or []:
            balance += sum(int(q or 0) for q in (event.get('shed') or {}).values())
            peak = max(peak, balance)
    return min(100, max(0, peak))


def protect_next_dawn_capacity(agent, obs: dict, action: dict, *, ordered=False) -> dict | None:
    day, hour = int(obs.get("day") or 0), int(obs.get("hour") or 0)
    if day >= 29 or hour != 23 or agent.preselected_edge is None:
        return None
    next_tape = agent.tapes.day_tape(str(agent.preselected_edge.get("example") or ""), day + 1)
    reserve = (next_dawn_place_room_ordered(next_tape) if ordered else next_dawn_place_room(next_tape))
    if not reserve:
        return None
    projected = _post_unit_private(obs, action)
    shed = {str(k): int(v or 0) for k, v in (projected.get("shed") or {}).items()}
    bags = sum(max(0, int(v or 0)) for bag in projected.get("inventories") or []
               for v in (bag or {}).values())
    for order in action.get("market") or []:
        if not order or len(order) < 3:
            continue
        op, item, qty = str(order[0]), str(order[1]), max(0, int(order[2] or 0))
        if op == "SELL":
            shed[item] = max(0, int(shed.get(item) or 0) - qty)
        elif op in ("BUY_PRODUCT", "BUY_ANIMAL"):
            shed[item] = int(shed.get(item) or 0) + qty
    before = sum(shed.values()) + bags
    need = max(0, before - (100 - reserve))
    if not need:
        return None
    prices = (obs.get("market") or {}).get("prices") or {}
    candidates = sorted((item for item in SELLABLE if item not in ("WHEAT", "FERTILIZER")),
                        key=lambda item: (-float(prices.get(item) or 0), item))
    sold = []
    for item in candidates:
        if need <= 0:
            break
        available = max(0, int(shed.get(item) or 0))
        if not available:
            continue
        existing = next((order for order in action.get("market") or []
                         if order and len(order) >= 3 and order[0] == "SELL" and order[1] == item), None)
        if existing is None and len(action.get("market") or []) >= 10:
            continue
        take = min(available, need)
        if existing is not None:
            existing[2] = int(existing[2]) + take
        else:
            action.setdefault("market", []).append(["SELL", item, take])
        shed[item] -= take
        need -= take
        sold.append({"item": item, "quantity": take})
    if sold:
        agent.market_seller.own_last_sales = {
            str(order[1]): int(order[2]) for order in action.get("market") or []
            if order and len(order) >= 3 and order[0] == "SELL"
        }
    result = {"day": day, "next_day": day + 1, "reserve_slots": reserve,
              "before_shed_plus_bags_after_original_market": before,
              "extra_sales": sold, "unresolved_slots": need}
    agent.choice.setdefault("experimental_next_dawn_capacity_guards", []).append(result)
    return result
