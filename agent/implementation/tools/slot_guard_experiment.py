"""Experimental seller-only protection for next dawn's ten market slots.

This is deliberately outside runtime/grid_agent.py while the node-RL runs have
that file hash-locked. It only trims a current SELL which would create an
extra next-hour BUY competing with source HIRE/BUY_LAND orders. It never adds
or reschedules a worker, procurement, or hire order.
"""
from __future__ import annotations

import copy

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from runtime.grid_agent import _post_unit_private, plan_source_hire_schedule
from runtime.resource_contract import contract_order_gaps, dawn_requirements
from runtime.tape_simplify import remove_dead_animal_pickups, trim_consumable_pickups


def protect_next_dawn_slots(agent, obs: dict, action: dict) -> dict | None:
    day, hour = int(obs.get("day") or 0), int(obs.get("hour") or 0)
    if hour != 23 or day >= 29 or agent.preselected_edge is None:
        return None
    next_tape = agent.tapes.day_tape(str(agent.preselected_edge.get("example") or ""), day + 1)
    if not next_tape:
        return None
    remove_dead_animal_pickups(next_tape)
    trim_consumable_pickups(next_tape)
    contract = dawn_requirements(next_tape[:2], rely_on_intraday_buys=False)
    projected = _post_unit_private(obs, action)
    # Nominal full fills are diagnostic only. In the actual game a rejected
    # order can still cause a shortage; this guard does not mask that failure.
    for order in action.get("market") or []:
        if not order or len(order) < 3:
            continue
        op, item, quantity = str(order[0]), str(order[1]), int(order[2] or 0)
        bucket = "seeds" if op == "BUY_SEED" else "shed"
        if op not in ("SELL", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_SEED"):
            continue
        stock = projected.setdefault(bucket, {})
        stock[item] = int(stock.get(item) or 0) + (-quantity if op == "SELL" else quantity)
    engine._drop_inventories_to_shed(projected, 100)
    farm = (obs.get("farms") or [{}, {}])[int(obs.get("player") or 0)]
    board_size = len(farm.get("tiles") or []) or 10
    hires = sum(int(value) == 0 for value in
                plan_source_hire_schedule(next_tape, board_size).get("hours") or [])
    land_target = int(next_tape[0].get("post_land") or 0)
    current_land = len(farm.get("unlocked_quadrants") or [])
    current_land += sum(bool(order and order[0] == "BUY_LAND")
                        for order in action.get("market") or [])
    lands = max(0, land_target - current_land)
    before_gaps = contract_order_gaps(contract, projected)[0]
    excess = max(0, hires + lands + len(before_gaps) - 10)
    trimmed = []
    if excess:
        for order in list(action.get("market") or []):
            if excess <= 0 or len(order) < 3 or order[0] != "SELL":
                continue
            item = str(order[1])
            # Only currently buyable inputs can cause the next market's
            # purchase-order count to exceed the cap in this contract.
            if item not in ("WHEAT", "FERTILIZER"):
                continue
            needed = int((contract.get("shed") or {}).get(item) or 0)
            held = int((projected.get("shed") or {}).get(item) or 0)
            if held >= needed:
                continue
            restore = min(int(order[2]), needed - held)
            order[2] -= restore
            projected["shed"][item] += restore
            trimmed.append({"item": item, "quantity": restore})
            if order[2] <= 0:
                action["market"].remove(order)
            excess = max(0, hires + lands + len(contract_order_gaps(contract, projected)[0]) - 10)
    if trimmed:
        agent.market_seller.own_last_sales = {
            str(order[1]): int(order[2]) for order in action.get("market") or []
            if order and order[0] == "SELL" and len(order) >= 3
        }
    result = {"day": day, "hour": hour, "next_day": day + 1,
              "hires": hires, "lands": lands, "before_gaps": before_gaps,
              "after_gaps": contract_order_gaps(contract, projected)[0],
              "trimmed_sales": trimmed, "unresolved_excess_orders": excess}
    agent.choice.setdefault("experimental_next_dawn_slot_guards", []).append(copy.deepcopy(result))
    return result
