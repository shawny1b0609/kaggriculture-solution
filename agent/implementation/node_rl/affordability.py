"""Candidate-specific next-day financing signal from an executable source tape.

This is an observable estimate, not a replacement for the runtime's timed
market orders. It intentionally excludes future shops, trades and opponent
state that are not visible at the branch decision.
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import SELLABLE
from runtime.grid_agent import plan_source_hire_schedule
from runtime.resource_contract import dawn_requirements


def next_day_contract(tape: list[dict]) -> dict:
    if not tape:
        raise ValueError("missing next-day source tape")
    stock = dawn_requirements(tape, rely_on_intraday_buys=False)
    hands = int(plan_source_hire_schedule(tape)["required_hands"])
    initial_land = int(tape[0].get("land") or 1)
    final_land = max(int(step.get("post_land") or step.get("land") or initial_land)
                     for step in tape)
    land_steps = max(0, final_land - initial_land)
    land_start = max(0, initial_land - 1)
    if land_start + land_steps > len(engine.LAND_PRICES):
        raise ValueError("source tape asks for unavailable land")
    return {
        "hire_count": hands,
        "hire_cost": sum(float(engine._hire_cost(index)) for index in range(hands)),
        "land_count": land_steps,
        "land_cost": sum(float(engine.LAND_PRICES[index])
                         for index in range(land_start, land_start + land_steps)),
        "seeds": stock["seeds"],
        "shed": stock["shed"],
    }


def affordability_vector(contract: dict, ctx: dict) -> list[float]:
    held_seeds = ctx.get("seeds") or {}
    held_shed = ctx.get("shed") or {}
    prices = ctx.get("market_prices") or {}
    required = float(contract["hire_cost"]) + float(contract["land_cost"])
    orders = int(contract["hire_count"]) + int(contract["land_count"])
    for item, need in contract["seeds"].items():
        gap = max(0, int(need) - int(held_seeds.get(item) or 0))
        if gap:
            orders += 1
            required += gap * float(engine.CROPS[item]["seed"])
    for item, need in contract["shed"].items():
        gap = max(0, int(need) - int(held_shed.get(item) or 0))
        if not gap:
            continue
        orders += 1
        unit_cost = (float(engine.ANIMALS[item]["cost"]) if item in engine.ANIMALS
                     else float(prices.get(item) or engine.MARKET_PARAMS[item]["base"]))
        required += gap * unit_cost
    liquidatable = 0.0
    for item in SELLABLE:
        # WHEAT/FERTILIZER needed for production are not free to sell.
        free = max(0, int(held_shed.get(item) or 0) - int(contract["shed"].get(item) or 0))
        liquidatable += free * float(prices.get(item) or engine.MARKET_PARAMS[item]["base"])
    cash = max(0.0, float(ctx.get("cash") or 0))
    # Market depth and order slots can lower realized proceeds; 70% is an
    # intentionally conservative, fixed estimate for the feature only.
    shortfall = max(0.0, required - cash - 0.7 * liquidatable)
    return [min(3.0, required / 10000.0),
            min(3.0, shortfall / 10000.0),
            min(2.0, orders / 10.0)]
