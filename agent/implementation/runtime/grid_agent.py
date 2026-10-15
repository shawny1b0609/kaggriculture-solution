"""On a matching dawn grid, replay that edge's featured worker dispatch.

Shopping is planned from the dest occupancy gap. Hires follow the source tape.
If the live grid is not on the net, do not splice another game's schedule.
"""
from __future__ import annotations

import copy
import random

import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import ANIMALS, CROPS, GRID_MAX_CHOICES, SELLABLE
from mining.features import encode_dest_branch, encode_obs_from_env
from mining.grid_key import (exact_cell, exact_hash, is_idle, is_weed, materials_needed,
                             prod_match, production_cell_match)
from runtime.bag_delivery import plan_early_deposit
from runtime.cash_repair import handoff_missed_maintenance, postpone_unfunded_plants
from runtime.entry import sample_program
from runtime.grid_store import GridNet
from runtime.node_policy import NodeScorer, pack_branches
from runtime.resource_contract import (advance_purchase_horizon, contract_order_gaps,
                                       dawn_requirements, first_stock_use_offsets,
                                       top_up_orders)
from runtime.seller import RollingLiquiditySeller, sell, sell_immediate
from runtime.source_tape import TapeBank
from runtime.tape_simplify import remove_dead_animal_pickups, trim_consumable_pickups
from runtime.view import own_farm
from runtime.weed_repair import (apply_idle_digs, plan_deferred_extra_hand,
                                 plan_extra_hand_targets, repair_deferred_jobs,
                                 repair_first_arrival_weeds, repair_same_hour_build_place,
                                 repair_same_hour_field_dependencies,
                                 repair_same_hour_plant_water)


def _post_unit_private(obs: dict, step: dict) -> dict:
    """Project own private stock at this turn's market phase."""
    farm = copy.deepcopy(own_farm(obs))
    private = copy.deepcopy(obs.get("private") or {})
    units = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
    demand = {}
    for action in units:
        if isinstance(action, list) and len(action) >= 2 and action[0] == "PLANT":
            demand[action[1]] = demand.get(action[1], 0) + 1
    blocked = {crop for crop, n in demand.items() if n > int((private.get("seeds") or {}).get(crop) or 0)}
    for idx, action in enumerate(units):
        if isinstance(action, list) and len(action) >= 2 and action[0] == "PLANT" and action[1] in blocked:
            continue
        engine._apply_unit_action(farm, private, idx, action, len(farm["tiles"]), int(obs.get("day") or 0), 24, 100)
    return private


def _post_unit_shed(obs: dict, step: dict) -> dict[str, int]:
    return {item: int(n or 0) for item, n in (_post_unit_private(obs, step).get("shed") or {}).items()}


def _terminal_drop_idle(obs: dict, farmer: list, hands: list[list]) -> tuple[list, list[list]]:
    """Deposit sellable bags on the final turn when a shed-side unit can.

    DROP discards overflow in the official engine. Only use it if the complete
    bag fits before the market phase, and never replace a productive action.
    A final pure move has no future production purpose and may be replaced.
    """
    farm = own_farm(obs)
    private = obs.get("private") or {}
    positions = [farm.get("farmer") or [], *(farm.get("hands") or [])]
    units = [list(farmer), *[list(hand) for hand in hands]]
    bags = private.get("inventories") or []
    room = 100 - sum(int(n or 0) for n in (private.get("shed") or {}).values())
    for idx, action in enumerate(units):
        if (action != ["PASS"] and (not action or action[0] not in ("NORTH", "SOUTH", "WEST", "EAST"))) or idx >= len(positions) or idx >= len(bags):
            continue
        bag = bags[idx] or {}
        total = sum(int(n or 0) for n in bag.values())
        if total <= 0 or total > room or not any(int(bag.get(item) or 0) > 0 for item in SELLABLE):
            continue
        if engine._is_shed_adjacent(tuple(positions[idx]), len(farm.get("tiles") or [])):
            units[idx] = ["DROP"]
            room -= total
    return units[0], units[1:]


def _extra_hire_affordable(obs: dict, source_market: list[list], reserve: float = 50) -> tuple[bool, float, float]:
    """Conservative same-hour cash test; source SELL proceeds are ignored."""
    farm = own_farm(obs)
    cash = float(farm.get("money") or 0)
    hires = int(farm.get("hires_today") or 0)
    land = max(0, len(farm.get("unlocked_quadrants") or []) - 1)
    market = obs.get("market") or {}
    inventory = market.get("inventory") or {}
    params = market.get("params")
    spent = 0.0
    for order in source_market:
        if not order:
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) >= 2 else ""
        qty = max(0, int(order[2] or 0)) if len(order) >= 3 else 1
        if op == "HIRE":
            spent += engine._hire_cost(hires)
            hires += 1
        elif op == "BUY_LAND" and land < len(engine.LAND_PRICES):
            spent += engine.LAND_PRICES[land]
            land += 1
        elif op == "BUY_SEED" and item in engine.CROPS:
            spent += qty * engine.CROPS[item]["seed"]
        elif op == "BUY_ANIMAL" and item in engine.ANIMALS:
            spent += qty * engine.ANIMALS[item]["cost"]
        elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
            initial = int(inventory.get(item) or 0)
            spent += sum(engine.market_price(item, initial - 1 - i, params) for i in range(qty))
    extra_cost = float(engine._hire_cost(hires))
    remaining = cash - spent
    return remaining >= extra_cost + reserve, remaining, extra_cost


def _estimate_market_cost(obs: dict, orders: list[list]) -> float:
    """Estimate own sequential BUY/HIRE/LAND cost before opponent slippage."""
    farm = own_farm(obs)
    hires = int(farm.get("hires_today") or 0)
    land = max(0, len(farm.get("unlocked_quadrants") or []) - 1)
    market = obs.get("market") or {}
    inventory = {str(k): int(v or 0) for k, v in (market.get("inventory") or {}).items()}
    params = market.get("params")
    cost = 0.0
    for order in orders:
        if not order:
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) >= 2 else ""
        quantity = max(0, int(order[2] or 0)) if len(order) >= 3 else 1
        if op == "HIRE":
            cost += float(engine._hire_cost(hires))
            hires += 1
        elif op == "BUY_LAND" and land < len(engine.LAND_PRICES):
            cost += float(engine.LAND_PRICES[land])
            land += 1
        elif op == "BUY_SEED" and item in engine.CROPS:
            cost += quantity * float(engine.CROPS[item]["seed"])
        elif op == "BUY_ANIMAL" and item in engine.ANIMALS:
            cost += quantity * float(engine.ANIMALS[item]["cost"])
        elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
            initial = int(inventory.get(item) or 0)
            cost += sum(float(engine.market_price(item, initial - 1 - i, params))
                        for i in range(quantity))
            inventory[item] = initial - quantity
    return cost


def _estimate_sale_value(obs: dict, orders: list[list]) -> float:
    market = obs.get("market") or {}
    inventory = {str(k): int(v or 0) for k, v in (market.get("inventory") or {}).items()}
    params = market.get("params")
    value = 0.0
    for order in orders:
        if not order or order[0] != "SELL" or len(order) < 3:
            continue
        item, quantity = str(order[1]), max(0, int(order[2] or 0))
        initial = int(inventory.get(item) or 0)
        for _ in range(quantity):
            price = float(engine.market_price(item, initial, params))
            value += price
            if price > 1:
                initial += 1
        inventory[item] = initial
    return value


def _projected_sellable_arrivals(tape: list[dict]) -> dict[str, int]:
    """Known source-tape output still expected in the active planning window."""
    arrivals = {item: 0 for item in SELLABLE}
    for step in tape or []:
        for item, quantity in (step.get("yielded") or {}).items():
            if item in arrivals:
                arrivals[item] += max(0, int(quantity or 0))
    return arrivals


def _prod_match_except(live: dict, dest: dict, ignored: set[tuple[int, int]]) -> bool:
    """Production-state match with a narrow, explicitly tracked recovery mask."""
    if sorted(live.get("unlocked_quadrants") or []) != sorted(dest.get("unlocked_quadrants") or []):
        return False
    lt, dt = live.get("tiles") or [], dest.get("tiles") or []
    if len(lt) != len(dt):
        return False
    for y, (lrow, drow) in enumerate(zip(lt, dt)):
        if len(lrow or []) != len(drow or []):
            return False
        for x, (lv, dv) in enumerate(zip(lrow or [], drow or [])):
            if (x, y) in ignored:
                continue
            if is_idle(dv) and is_idle(lv):
                continue
            if not production_cell_match(lv, dv):
                return False
    return True


def _field_job_present(tile, job: list) -> bool:
    if not isinstance(tile, dict) or not job:
        return False
    if job[0] == "PLANT":
        return len(job) >= 2 and tile.get("kind") == "PLANT" and tile.get("crop") == job[1]
    if job[0] == "BUILD_COOP":
        return tile.get("kind") == "COOP"
    if job[0] == "BUILD_PASTURE":
        return tile.get("kind") == "PASTURE"
    return False


def _cap_daily_buys(market: list[list], projected: dict, future_tape: list[dict]) -> list[list]:
    """Keep source purchase timing, but only for this day's unit demand."""
    need = dawn_requirements(future_tape, rely_on_intraday_buys=False)
    held = {
        "seeds": dict(projected.get("seeds") or {}),
        "shed": dict(projected.get("shed") or {}),
    }
    result = []
    for order in market:
        if len(order) < 3 or order[0] not in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"):
            result.append(order)
            continue
        op, item = str(order[0]), str(order[1])
        bucket = "seeds" if op == "BUY_SEED" else "shed"
        have = int(held[bucket].get(item) or 0)
        cap = max(0, int((need[bucket]).get(item) or 0) - have)
        qty = min(max(0, int(order[2] or 0)), cap)
        if qty > 0:
            result.append([op, item, qty])
            held[bucket][item] = have + qty
    return result


def _cap_structural_orders(market: list[list], step: dict, farm: dict, hour: int) -> list[list]:
    """Reissue source HIRE/LAND only up to its observed post-market state.

    Featured agents often request more hires or land than their cash actually
    buys. Replaying the request count with different cash creates additional
    hands/quadrants and invalidates the witness worker indices or farm grid.
    """
    target_hands = step.get("post_hands")
    target_land = step.get("post_land")
    hires_left = None if target_hands is None else max(0, int(target_hands) - len(farm.get("hands") or []))
    if hour == 23:
        # End-of-day resets hands before another field action can use them.
        hires_left = 0
    land_left = None if target_land is None else max(0, int(target_land) - len(farm.get("unlocked_quadrants") or []))
    result = []
    for order in market:
        if order[0] == "HIRE" and hires_left is not None:
            if hires_left <= 0:
                continue
            hires_left -= 1
        elif order[0] == "BUY_LAND" and land_left is not None:
            if land_left <= 0:
                continue
            land_left -= 1
        result.append(order)
    return result


def _tape_cover(tape: list[dict]) -> dict[tuple, int]:
    cover: dict[tuple, int] = {}
    for step in tape or []:
        for order in step.get("market") or []:
            if not order:
                continue
            op = str(order[0])
            if op == "BUY_LAND":
                cover[("BUY_LAND", "")] = cover.get(("BUY_LAND", ""), 0) + 1
            elif op in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and len(order) >= 3:
                key = (op, str(order[1]))
                cover[key] = cover.get(key, 0) + int(order[2] or 0)
    return cover


def shop_gaps(needed: list[list], tape: list[dict]) -> list[list]:
    """Only buy what the source tape will not already buy today."""
    cover = _tape_cover(tape)
    out = []
    for order in needed or []:
        if not order:
            continue
        op = str(order[0])
        if op == "BUY_LAND":
            key = ("BUY_LAND", "")
            have = cover.get(key, 0)
            if have > 0:
                cover[key] = have - 1
                continue
            out.append(list(order))
            continue
        if op in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and len(order) >= 3:
            key = (op, str(order[1]))
            have = cover.get(key, 0)
            want = int(order[2] or 0)
            if have >= want:
                cover[key] = have - want
                continue
            gap = want - have
            cover[key] = 0
            out.append([op, order[1], gap])
            continue
        out.append(list(order))
    return out


def occup_match(stats: dict, node: dict | None) -> bool:
    if not node:
        return False
    dest_c = node.get("crops") or {}
    dest_a = node.get("animals") or {}
    live_c = stats.get("crops") or {}
    live_a = stats.get("animals") or {}
    if int(node.get("land") or 1) != int(stats.get("land") or 1):
        return False
    for name in CROPS:
        if int(live_c.get(name) or 0) != int(dest_c.get(name) or 0):
            return False
    for name in ANIMALS:
        if int(live_a.get(name) or 0) != int(dest_a.get(name) or 0):
            return False
    return True


def _pass_action(obs) -> dict:
    hands = own_farm(obs).get("hands") or []
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}


def plan_source_hire_schedule(tape: list[dict], board_size: int = 10) -> dict:
    """Recover the source's *fulfilled* hire cadence from worker births.

    Worker indices and shed-access spawn tiles depend on every earlier hire,
    including an otherwise idle hand.  Replaying all observed births at their
    original market hour is therefore safer than delaying apparently idle
    workers.  Dedicated weed hands are planned separately and appended only
    after this complete source prefix.
    """
    del board_size  # Kept in the public signature for existing callers.
    required_hands = max(
        (max(0, len(step.get("positions") or []) - 1) for step in (tape or [])),
        default=0,
    )
    hours = []
    missing_births = []
    for hand_index in range(required_hands):
        birth = next(
            (hour for hour, step in enumerate(tape or [])
             if len(step.get("positions") or []) > hand_index + 1),
            None,
        )
        if birth is None:
            missing_births.append(hand_index)
            continue
        hours.append(max(0, int(birth) - 1))
    return {
        "hours": hours,
        "original_hours": list(hours),
        "required_hands": required_hands,
        "fallback": bool(missing_births),
        "fallback_reason": "missing_source_birth" if missing_births else None,
        "missing_births": missing_births,
        "saved_hire_hours": 0,
        "calibrations": [],
    }


def next_day_staffing_reserve(tape: list[dict], farm: dict) -> float:
    """Cash that must exist before a selected next-day worker tape starts.

    Hires reset every dawn, so a day whose first market already uses all ten
    order slots cannot finance itself with a same-turn sale.  The preceding
    h23 market therefore protects the complete next-day hire bill plus any
    additional land required by that tape.
    """
    plan = plan_source_hire_schedule(tape)
    hires = int(plan.get("required_hands") or 0)
    reserve = sum(float(engine._hire_cost(index)) for index in range(hires))
    current_land = len(farm.get("unlocked_quadrants") or [])
    target_land = max(
        (int(step.get("post_land") or step.get("land") or current_land) for step in tape or []),
        default=current_land,
    )
    extra_land = max(0, target_land - current_land)
    land_index = max(0, current_land - 1)
    for offset in range(extra_land):
        index = land_index + offset
        if index < len(engine.LAND_PRICES):
            reserve += float(engine.LAND_PRICES[index])
    return reserve


class GridAgent:
    def __init__(
        self,
        net: GridNet,
        policy: NodeScorer | None = None,
        seed: int = 0,
        greedy: bool = True,
        device: str = "cpu",
        root_id: int | None = None,
        tapes: TapeBank | None = None,
        path: list | None = None,
        follow: str = "greedy",
        sell_mode: str = "immediate",
        repair_weeds: bool = True,
        repair_cash: bool = True,
        market_mode: str = "derived",
    ):
        self.net = net
        self.policy = policy
        self.device = device
        self.greedy = greedy
        self.follow = follow
        if sell_mode not in ("seller", "immediate", "source"):
            raise ValueError(f"unknown sell mode: {sell_mode}")
        self.sell_mode = "source" if market_mode == "raw_source" else sell_mode
        if market_mode not in ("raw_source", "derived", "inverse", "planned", "daily", "source"):
            raise ValueError(f"unknown market mode: {market_mode}")
        self.market_mode = market_mode
        self.repair_weeds = repair_weeds and market_mode != "raw_source"
        self.repair_cash = repair_cash and market_mode == "derived"
        self.rng = random.Random(seed)
        self.tapes = tapes or TapeBank()
        self.node_id = int(net.day0_id())
        self.root_id = int(root_id) if root_id is not None else None
        self._opened = False
        self.fixed_path = list(path or [])
        self.path_i = 0
        self.pending = None
        self.preselected_edge: dict | None = None
        self.loc = ""
        self.tape: list[dict] = []
        self.weed_plan: dict[tuple[int, int], tuple[int, int]] = {}
        self.extra_hand_plan: dict | None = None
        self.deferred_weed_jobs: list[dict] = []
        self.deferred_cash_jobs: list[dict] = []
        self.pending_weed_jobs: list[dict] = []
        self.pending_seed_order: dict | None = None
        self.market_backlog: dict[tuple[str, str], dict] = {}
        self.market_backlog_seq = 0
        # Capital formation uses the seller's four-turn bootstrap horizon;
        # after working cash reaches 10k, inspect two days of known demand.
        self.market_seller = RollingLiquiditySeller(price_window_turns=48)
        self.market_seller.bootstrap_window_turns = 4
        self.market_capital_floor = 0.0
        self.hire_plan_day = -1
        self.hire_plan: dict = {"hours": [], "original_hours": [], "required_hands": 0}
        self.calls = 0
        self.misses = 0
        self.hits = 0
        self.shop_days = 0
        self.choice = {
            "root": self.root_id,
            "path": [],
            "forks": [],
            "misses": 0,
            "hits": 0,
            "shop_days": 0,
            "tapes": 0,
            "prod_hits": 0,
            "exact_miss_prod_hit": 0,
            "weed_repairs_planned": 0,
            "weed_repairs_used": 0,
            "weed_routes_changed": 0,
            "weed_delays": 0,
            "weed_first_arrival_repairs": 0,
            "weed_first_arrival_details": [],
            "weed_dependency_swaps": 0,
            "weed_dependency_delays": [],
            "weed_uncovered": [],
            "weed_extra_planned": 0,
            "weed_extra_requested": 0,
            "weed_extra_skipped": [],
            "weed_extra_deferred": [],
            "weed_extra_liquidity_obligations": [],
            "weed_extra_handoffs": [],
            "weed_deferred": [],
            "weed_deferred_recovery": [],
            "weed_deferred_rejoined": [],
            "weed_deferred_cleared": [],
            "weed_deferred_hard_failures": [],
            "resource_prep": [],
            "resource_shortages": [],
            "market_seed_shortages": [],
            "market_order_shortages": [],
            "market_backlog_added": [],
            "market_backlog_filled": [],
            "market_backlog_deferred": [],
            "market_backlog_overdue": [],
            "market_hard_bundles": [],
            "market_impossible_bundles": [],
            "market_optional_buys": [],
            "market_sell_plans": [],
            "market_sell_accounting": {
                "planned_orders": 0, "issued_orders": 0,
                "planned_units": 0, "issued_units": 0,
                "planned_stressed_revenue_estimate": 0.0,
                "issued_stressed_revenue_estimate": 0.0,
            },
            "cash_by_day": [],
            "hire_schedule_plans": [],
            "hire_schedule_shortages": [],
            "dead_animal_pickups_removed": [],
            "consumable_pickups_trimmed": [],
            "early_deposits": [],
            "cash_seed_delays": [],
            "cash_seed_unresolved": [],
            "cash_local_deferred": [],
            "cash_local_tolerated_dawns": [],
            "cash_local_recovery": [],
            "cash_local_rejoined": [],
            "cash_local_blocked": [],
            "cash_maintenance_handoffs": [],
        }
        self.trajectory: list[dict] = []
        self.program_id = "grid_tape"
        self._last_day = -1
        if policy is not None:
            policy.eval()

    def __call__(self, obs, cfg=None):
        self.calls += 1
        self._check_seed_order_fill(obs)
        day = int(obs.get("day") or 0)
        hour = int(obs.get("hour") or 0)
        if hour == 0 and day != self._last_day:
            self._dawn(obs, day)
            self._last_day = day
        if self.tape:
            action = self._play(obs, hour)
            self._remember_seed_order(obs, action)
            return action
        action = _pass_action(obs)
        if day * 24 + hour == 718 and self.sell_mode != "source":
            action["farmer"], action["hands"] = _terminal_drop_idle(obs, action["farmer"], action["hands"])
        if self.market_mode in ("inverse", "planned"):
            action["market"] = sell_immediate(_post_unit_shed(obs, action), {})
        elif self.sell_mode == "immediate":
            action["market"] = sell_immediate(_post_unit_shed(obs, action), {})
        elif day * 24 + hour == 718 and self.sell_mode == "seller":
            action["market"] = sell_immediate(_post_unit_shed(obs, action), {})
        self._remember_seed_order(obs, action)
        return action

    def _choice_log(self, key: str, row: dict, limit: int = 256) -> None:
        rows = self.choice.setdefault(key, [])
        if len(rows) < limit:
            rows.append(row)

    def _defer_extra_hire(self, day: int, hour: int, reason: str) -> None:
        if self.extra_hand_plan is None:
            return
        self.extra_hand_plan["requested"] = False
        if hour < 22:
            self.extra_hand_plan["hire_hour"] = hour + 1
        self._choice_log("weed_extra_deferred", {
            "day": day, "hour": hour, "retry_hour": self.extra_hand_plan.get("hire_hour"),
            "reason": reason,
        })

    def _queue_market_order(self, op: str, item: str, quantity: int, *,
                            due_turn: int, priority: int = 99, reason: str = "contract",
                            planned_turn: int | None = None,
                            hard_quantity: int = 0) -> None:
        """Create or refresh one persistent warehouse purchase intent."""
        quantity = max(0, int(quantity or 0))
        if quantity <= 0:
            return
        planned_turn = int(due_turn if planned_turn is None else planned_turn)
        hard_quantity = min(quantity, max(0, int(hard_quantity or 0)))
        key = (str(op), str(item))
        entry = self.market_backlog.get(key)
        if entry is None:
            self.market_backlog_seq += 1
            entry = {
                "op": key[0], "item": key[1], "remaining": quantity,
                "due_turn": int(due_turn), "priority": int(priority),
                "seq": self.market_backlog_seq, "reason": str(reason),
                "last_submitted_turn": None,
                "last_planned_turn": planned_turn,
                "current_priority": int(priority),
                "current_quantity": quantity,
                "current_hard_quantity": hard_quantity,
            }
            self.market_backlog[key] = entry
            self._choice_log("market_backlog_added", dict(entry))
            return
        old = int(entry.get("remaining") or 0)
        entry["remaining"] = max(old, quantity)
        old_due = int(entry["due_turn"]) if entry.get("due_turn") is not None else int(due_turn)
        old_priority = int(entry["priority"]) if entry.get("priority") is not None else int(priority)
        entry["due_turn"] = min(old_due, int(due_turn))
        entry["priority"] = min(old_priority, int(priority))
        entry["last_planned_turn"] = planned_turn
        entry["current_priority"] = int(priority)
        entry["current_quantity"] = quantity
        entry["current_hard_quantity"] = hard_quantity
        entry["current_reason"] = str(reason)
        if int(entry["remaining"]) > old:
            self._choice_log("market_backlog_added", {
                "op": key[0], "item": key[1], "added": int(entry["remaining"]) - old,
                "remaining": int(entry["remaining"]), "due_turn": int(entry["due_turn"]),
                "reason": str(reason),
            })

    def _apply_market_backlog_fill(self, op: str, item: str, filled: int,
                                   *, day: int, hour: int) -> None:
        key = (str(op), str(item))
        entry = self.market_backlog.get(key)
        filled = max(0, int(filled or 0))
        if entry is None or filled <= 0:
            return
        used = min(filled, int(entry.get("remaining") or 0))
        entry["remaining"] = int(entry.get("remaining") or 0) - used
        self._choice_log("market_backlog_filled", {
            "day": int(day), "hour": int(hour), "op": key[0], "item": key[1],
            "filled": used, "remaining": int(entry["remaining"]),
        })
        if int(entry["remaining"]) <= 0:
            del self.market_backlog[key]

    def _take_market_backlog_orders(self, private: dict, *, max_orders: int,
                                    turn: int, shed_capacity: int = 100,
                                    respect_capacity: bool = True) -> list[list]:
        """Take the highest-priority queue prefix without deleting its intent."""
        max_orders = max(0, int(max_orders or 0))
        shed = private.get("shed") or {}
        room = max(0, int(shed_capacity) - sum(int(v or 0) for v in shed.values()))
        ordered = sorted(
            (entry for entry in self.market_backlog.values() if int(entry.get("remaining") or 0) > 0),
            key=lambda row: (int(row["due_turn"]) if row.get("due_turn") is not None else 0,
                             int(row["priority"]) if row.get("priority") is not None else 99,
                             int(row.get("seq") or 0)),
        )
        result = []
        selected: set[tuple[str, str]] = set()
        for entry in ordered:
            key = (str(entry["op"]), str(entry["item"]))
            if int(entry["due_turn"]) < int(turn):
                self._choice_log("market_backlog_overdue", {
                    "turn": int(turn), "op": key[0], "item": key[1],
                    "remaining": int(entry["remaining"]),
                    "due_turn": int(entry["due_turn"]),
                })
            if len(result) >= max_orders:
                continue
            if entry.get("last_submitted_turn") == int(turn):
                continue
            take = int(entry["remaining"])
            if respect_capacity and key[0] in ("BUY_ANIMAL", "BUY_PRODUCT"):
                take = min(take, room)
                if take <= 0:
                    self._choice_log("market_backlog_deferred", {
                        "turn": int(turn), "op": key[0], "item": key[1],
                        "remaining": int(entry["remaining"]), "reason": "shed_capacity",
                    })
                    continue
                room -= take
            result.append([key[0], key[1], take])
            selected.add(key)
            entry["last_submitted_turn"] = int(turn)
            entry["last_submitted_quantity"] = take
        if len(result) >= max_orders:
            for entry in ordered:
                key = (str(entry["op"]), str(entry["item"]))
                if key in selected or entry.get("last_submitted_turn") == int(turn):
                    continue
                self._choice_log("market_backlog_deferred", {
                    "turn": int(turn), "op": key[0], "item": key[1],
                    "remaining": int(entry["remaining"]), "reason": "market_order_limit",
                })
        return result

    def _market_backlog_groups(self, turn: int) -> tuple[list[list], list[list], list[list]]:
        """Split queued purchases into current hard, current future, and stale.

        A formerly overdue order remains recorded, but it cannot outrank a
        purchase recomputed for the current next field phase.
        """
        hard_rows = []
        future_rows = []
        stale_rows = []
        for entry in self.market_backlog.values():
            remaining = max(0, int(entry.get("remaining") or 0))
            if remaining <= 0:
                continue
            last_planned = entry.get("last_planned_turn")
            active = last_planned is not None and int(last_planned) == int(turn)
            current = min(remaining, max(0, int(entry.get("current_quantity") or 0))) if active else 0
            hard = min(current, max(0, int(entry.get("current_hard_quantity") or 0)))
            future = max(0, current - hard)
            stale = max(0, remaining - current) if active else remaining
            key = (str(entry["op"]), str(entry["item"]))
            if hard:
                hard_rows.append((int(entry.get("current_priority") or 0), int(entry.get("seq") or 0),
                                  [key[0], key[1], hard]))
            if future:
                future_rows.append((int(entry.get("current_priority") or 99), int(entry.get("seq") or 0),
                                    [key[0], key[1], future]))
            if stale:
                stale_rows.append((int(entry.get("due_turn") or 0), int(entry.get("seq") or 0),
                                   [key[0], key[1], stale]))
        hard = [row[2] for row in sorted(hard_rows)]
        future = [row[2] for row in sorted(future_rows)]
        stale = [row[2] for row in sorted(stale_rows)]
        return hard, future, stale

    def _mark_backlog_submitted(self, order: list, turn: int) -> None:
        if len(order) < 3:
            return
        entry = self.market_backlog.get((str(order[0]), str(order[1])))
        if entry is None:
            return
        entry["last_submitted_turn"] = int(turn)
        entry["last_submitted_quantity"] = max(0, int(order[2] or 0))

    def _ensure_hire_plan(self, day: int, board_size: int) -> None:
        if self.hire_plan_day == int(day):
            return
        plan = plan_source_hire_schedule(self.tape, board_size)
        self.hire_plan = plan
        self.hire_plan_day = int(day)
        self._choice_log("hire_schedule_plans", {
            "day": int(day), "loc": self.loc,
            "hours": list(plan.get("hours") or []),
            "original_hours": list(plan.get("original_hours") or []),
            "required_hands": int(plan.get("required_hands") or 0),
            "saved_hire_hours": int(plan.get("saved_hire_hours") or 0),
            "fallback": bool(plan.get("fallback")),
            "fallback_reason": plan.get("fallback_reason"),
            "calibrations": [],
        }, limit=64)

    def _future_source_staffing_orders(self, hour: int, farm: dict,
                                       current_staffing: list[list]) -> list[list]:
        """Structural cash that optional prebuys may not consume."""
        future_hires = sum(
            int(hire_hour) > int(hour)
            for hire_hour in (self.hire_plan.get("hours") or [])
        )
        land_after_current = (
            len(farm.get("unlocked_quadrants") or [])
            + sum(bool(order and order[0] == "BUY_LAND") for order in current_staffing)
        )
        future_land_target = max(
            (int(step.get("post_land") or step.get("land") or 1)
             for step in self.tape[hour + 1:]),
            default=land_after_current,
        )
        future_lands = max(0, future_land_target - land_after_current)
        return [["HIRE"] for _ in range(future_hires)] + [
            ["BUY_LAND"] for _ in range(future_lands)
        ]

    def _schedule_current_first_market(
        self,
        obs: dict,
        projected: dict,
        hard_buys: list[list],
        staffing: list[list],
        future_buys: list[list],
        stale_buys: list[list],
        sales: list[list],
        future_staffing: list[list],
        *,
        turn: int,
        day: int,
        hour: int,
        protected_cash_after_hard: float | None = None,
    ) -> tuple[list[list], list[list], list[list]]:
        """Reserve farm-order slots, then put sales into any spare slots first.

        Sales never displace planned hard orders. Their same-turn receipts and
        released shed room may support farm orders that follow them. Unissued
        purchases stay queued; source hires are recomputed next turn.
        """
        hard_bundle = [list(order) for order in hard_buys + staffing]
        # A blocked warehouse order must not hide a feasible source hire:
        # worker births determine all later indices and spawn tiles. Purchase
        # priorities remain in their first-stock-use order within the suffix.
        priority_bundle = [list(order) for order in staffing + hard_buys]
        cash0 = float(own_farm(obs).get("money") or 0)
        room0 = max(0, 100 - sum(int(v or 0) for v in (projected.get("shed") or {}).values()))

        def shed_need(orders: list[list]) -> int:
            return sum(max(0, int(order[2] or 0)) for order in orders
                       if len(order) >= 3 and order[0] in ("BUY_ANIMAL", "BUY_PRODUCT"))

        def sale_value(orders: list[list]) -> float:
            if self.sell_mode == "seller":
                return self.market_seller.stressed_revenue(obs, orders)
            return _estimate_sale_value(obs, orders)

        def sale_room(orders: list[list]) -> int:
            return sum(max(0, int(order[2] or 0)) for order in orders if len(order) >= 3)

        # Reserve one slot for every planned hard order, capped at ten. Find
        # the smallest prefix of the seller's orders that makes that reserved
        # bundle affordable and fits it in the shed; other sales are optional.
        hard_claim = priority_bundle[:10]
        max_sale_for_hard = max(0, 10 - len(hard_claim))
        needed_sales = 0
        for count in range(min(len(sales), max_sale_for_hard) + 1):
            prefix = sales[:count]
            if (_estimate_market_cost(obs, hard_claim) <= cash0 + sale_value(prefix) + 1e-9
                    and shed_need(hard_claim) <= room0 + sale_room(prefix)):
                needed_sales = count
                break
        else:
            needed_sales = min(len(sales), max_sale_for_hard)
        funding_sales = [list(order) for order in sales[:needed_sales]]
        cash_for_farm = cash0 + sale_value(funding_sales)

        active_hard: list[list] = []
        deferred_hard: list[dict] = []
        room = room0 + sale_room(funding_sales)
        for order in priority_bundle:
            reason = None
            quantity = max(0, int(order[2] or 0)) if len(order) >= 3 else 1
            if len(active_hard) >= 10:
                reason = "market_order_limit"
            elif order[0] in ("BUY_ANIMAL", "BUY_PRODUCT") and quantity > room:
                reason = "shed_capacity"
            elif _estimate_market_cost(obs, active_hard + [order]) > cash_for_farm + 1e-9:
                reason = "cash"
            if reason is not None:
                deferred_hard.append({"order": list(order), "reason": reason})
                continue
            active_hard.append(list(order))
            if order[0] in ("BUY_ANIMAL", "BUY_PRODUCT"):
                room -= quantity
        hard_complete = not deferred_hard
        self.last_market_hard_deferred = deferred_hard
        market = funding_sales + [list(order) for order in active_hard]
        issued_hard_buys = [order for order in active_hard
                            if order and order[0] in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT")]
        issued_staffing = [order for order in active_hard
                           if order and order[0] in ("HIRE", "BUY_LAND")]
        hard_cost = _estimate_market_cost(obs, active_hard)
        cash_after_hard = cash_for_farm - hard_cost
        room_after_hard = room0 + sale_room(funding_sales) - shed_need(active_hard)
        future_staffing_reserve = max(
            0.0,
            _estimate_market_cost(obs, staffing + future_staffing)
            - _estimate_market_cost(obs, staffing),
        )
        protected_cash = (
            future_staffing_reserve
            if protected_cash_after_hard is None
            else max(future_staffing_reserve, float(protected_cash_after_hard))
        )
        deferred_reason = deferred_hard[0]["reason"] if deferred_hard else None
        self._choice_log("market_hard_bundles", {
            "day": day, "hour": hour, "required": hard_bundle,
            "issued": active_hard, "front_sales": funding_sales, "deferred": deferred_hard,
            "deferred_reason": deferred_reason,
            "complete": bool(hard_complete),
            "cash_before": round(cash0, 3),
            "cash_after_hard_estimate": round(cash_after_hard, 3),
            "future_staffing_reserve": round(future_staffing_reserve, 3),
            "protected_cash_after_hard": round(protected_cash, 3),
        })
        if not hard_complete:
            self._choice_log("market_impossible_bundles", {
                "day": day, "hour": hour, "required": hard_bundle,
                "issued": active_hard, "deferred": deferred_hard,
                "reason": deferred_reason,
            })
            for row in deferred_hard:
                order = row["order"]
                self._choice_log("market_hard_deferred", {
                    "turn": turn, "op": str(order[0]),
                    "item": str(order[1]) if len(order) >= 2 else "",
                    "remaining": int(order[2]) if len(order) >= 3 else 1,
                    "reason": row["reason"],
                })

        # Optional prebuys get farm-order priority over discretionary sales.
        # They may use already-reserved front-sale receipts, but are not
        # allowed to consume the slots those receipts require.
        selected_optional = []
        issued_keys = {(str(order[0]), str(order[1])) for order in issued_hard_buys}
        if hard_complete:
            for kind, candidates in (("future", future_buys), ("stale", stale_buys)):
                for raw in candidates:
                    key = (str(raw[0]), str(raw[1]))
                    if key in issued_keys:
                        continue
                    if len(market) + len(selected_optional) >= 10:
                        self._choice_log("market_backlog_deferred", {
                            "turn": turn, "op": key[0], "item": key[1],
                            "remaining": int(raw[2] or 0), "reason": "market_order_limit",
                        })
                        continue
                    max_quantity = max(0, int(raw[2] or 0))
                    if key[0] in ("BUY_ANIMAL", "BUY_PRODUCT"):
                        max_quantity = min(max_quantity, max(0, room_after_hard))
                    chosen = None
                    liquidity_floor = protected_cash + (50.0 if kind == "stale" else 0.0)
                    for quantity in range(max_quantity, 0, -1):
                        candidate = [key[0], key[1], quantity]
                        context = active_hard + selected_optional
                        incremental = (
                            _estimate_market_cost(obs, context + [candidate])
                            - _estimate_market_cost(obs, context)
                        )
                        optional_spent = (
                            _estimate_market_cost(obs, active_hard + selected_optional)
                            - hard_cost
                        )
                        if cash_after_hard - optional_spent - incremental >= liquidity_floor - 1e-9:
                            chosen = candidate
                            break
                    if chosen is None:
                        self._choice_log("market_backlog_deferred", {
                            "turn": turn, "op": key[0], "item": key[1],
                            "remaining": int(raw[2] or 0),
                            "reason": ("shed_capacity" if max_quantity <= 0
                                       else "cash_reserved_for_future_staffing"),
                        })
                        continue
                    selected_optional.append(chosen)
                    issued_keys.add(key)
                    if key[0] in ("BUY_ANIMAL", "BUY_PRODUCT"):
                        room_after_hard -= int(chosen[2] or 0)
                    self._choice_log("market_optional_buys", {
                        "day": day, "hour": hour, "kind": kind,
                        "order": chosen, "future_staffing_reserve": round(future_staffing_reserve, 3),
                        "protected_cash_after_hard": round(protected_cash, 3),
                    })

        # A failed hard order still owns its reserved slot this turn. Do not
        # turn a full ten-order farm bundle into a sale merely because one
        # purchase could not be funded or fit in the shed.
        available_sale_slots = max(0, 10 - len(hard_claim) - len(selected_optional))
        actual_sales = [list(order) for order in sales[:available_sale_slots]]
        market = actual_sales + [list(order) for order in active_hard] + selected_optional
        issued_buys = issued_hard_buys + selected_optional
        for order in issued_buys:
            self._mark_backlog_submitted(order, turn)
        if len(market) > 10:
            raise ValueError(f"market has {len(market)} orders at hour {hour}")
        return market, issued_buys, issued_staffing

    def _check_seed_order_fill(self, obs: dict) -> None:
        prior = self.pending_seed_order
        self.pending_seed_order = None
        if prior is None:
            return
        private = obs.get("private") or {}
        seeds = private.get("seeds") or {}
        shed = private.get("shed") or {}
        farm = own_farm(obs)
        cash_next = farm.get("money")
        for item, row in prior["items"].items():
            filled = max(0, int(seeds.get(item) or 0) - row["before"])
            self._apply_market_backlog_fill(
                "BUY_SEED", item, filled, day=prior["day"], hour=prior["hour"],
            )
            if filled < row["requested"]:
                shortage = {
                    "day": prior["day"], "hour": prior["hour"], "loc": prior["loc"],
                    "item": item, "requested": row["requested"], "filled": filled,
                    "cash_before_market": prior["cash"],
                    "cash_next_observation": cash_next,
                }
                self.choice["market_seed_shortages"].append(shortage)
                self.choice["market_order_shortages"].append({"op": "BUY_SEED", **shortage})
        # Across a day boundary worker bags may also enter the shed.  That makes
        # the provenance of the increase ambiguous, but it still satisfies the
        # queued stock need, so count it toward backlog completion.
        for key, row in prior.get("shed_items", {}).items():
            filled = max(0, int(shed.get(row["item"]) or 0) - row["before"])
            self._apply_market_backlog_fill(
                row["op"], row["item"], filled, day=prior["day"], hour=prior["hour"],
            )
            if filled < row["requested"]:
                self.choice["market_order_shortages"].append({
                    "day": prior["day"], "hour": prior["hour"], "loc": prior["loc"],
                    "op": row["op"], "item": row["item"],
                    "requested": row["requested"], "filled": filled,
                    "cash_before_market": prior["cash"], "cash_next_observation": cash_next,
                })
        if prior["hour"] < 23:
            requested_hires = int(prior.get("hires") or 0)
            if requested_hires:
                filled = max(0, len(farm.get("hands") or []) - int(prior.get("hands_before") or 0))
                if filled < requested_hires:
                    self.choice["market_order_shortages"].append({
                        "day": prior["day"], "hour": prior["hour"], "loc": prior["loc"],
                        "op": "HIRE", "requested": requested_hires, "filled": filled,
                        "cash_before_market": prior["cash"], "cash_next_observation": cash_next,
                    })
                    if prior.get("extra_hire_requested"):
                        self._defer_extra_hire(prior["day"], prior["hour"], "engine_hire_unfilled")
            requested_land = int(prior.get("land") or 0)
            if requested_land:
                filled = max(0, len(farm.get("unlocked_quadrants") or []) - int(prior.get("land_before") or 0))
                if filled < requested_land:
                    self.choice["market_order_shortages"].append({
                        "day": prior["day"], "hour": prior["hour"], "loc": prior["loc"],
                        "op": "BUY_LAND", "requested": requested_land, "filled": filled,
                        "cash_before_market": prior["cash"], "cash_next_observation": cash_next,
                    })

    def _remember_seed_order(self, obs: dict, action: dict) -> None:
        quantities = {}
        shed_quantities = {}
        sold_items = set()
        hires = 0
        land = 0
        for order in action.get("market") or []:
            if not isinstance(order, list) or not order:
                continue
            op = str(order[0])
            if op == "HIRE":
                hires += 1
            elif op == "BUY_LAND":
                land += 1
            elif len(order) >= 3 and op == "BUY_SEED":
                item = str(order[1])
                quantities[item] = quantities.get(item, 0) + max(0, int(order[2] or 0))
            elif len(order) >= 3 and op in ("BUY_ANIMAL", "BUY_PRODUCT"):
                item = str(order[1])
                key = (op, item)
                shed_quantities[key] = shed_quantities.get(key, 0) + max(0, int(order[2] or 0))
            elif len(order) >= 3 and op == "SELL":
                sold_items.add(str(order[1]))
        if not quantities and not shed_quantities and not hires and not land:
            return
        projected = _post_unit_private(obs, action)
        seeds = projected.get("seeds") or {}
        shed = projected.get("shed") or {}
        farm = own_farm(obs)
        self.pending_seed_order = {
            "day": int(obs.get("day") or 0), "hour": int(obs.get("hour") or 0), "loc": self.loc,
            "cash": farm.get("money"),
            "extra_hire_requested": bool(
                self.extra_hand_plan and self.extra_hand_plan.get("requested")
                and int(self.extra_hand_plan.get("hire_hour", -1)) == int(obs.get("hour") or 0)
                and hires
            ),
            "items": {item: {"before": int(seeds.get(item) or 0), "requested": n}
                      for item, n in quantities.items()},
            "shed_items": {
                f"{op}:{item}": {"op": op, "item": item, "before": int(shed.get(item) or 0),
                                  "requested": n}
                for (op, item), n in shed_quantities.items() if item not in sold_items
            },
            "hires": hires,
            "hands_before": len(farm.get("hands") or []),
            "land": land,
            "land_before": len(farm.get("unlocked_quadrants") or []),
        }

    def _dawn(self, obs, day: int):
        """Day 0 picks a root tape. Days 1–29 are nodes; day 29 is the last choice.

        A switch is allowed at a production-matched aimed node or an exact
        alternative node. Never continue a stale source schedule off-graph.
        """
        if day > 29:
            self.tape = []
            return
        farm = own_farm(obs)
        self.choice["cash_by_day"].append({"day": int(day), "money": float(farm.get("money") or 0)})
        if self.pending_weed_jobs:
            known = {tuple(int(v) for v in row["xy"]) for row in self.deferred_weed_jobs}
            promoted = []
            for target in self.pending_weed_jobs:
                xy = tuple(int(v) for v in target["xy"])
                job = list(target.get("job") or [])
                tile = farm["tiles"][xy[1]][xy[0]]
                if xy in known or _field_job_present(tile, job):
                    continue
                row = {"xy": list(xy), "job": job, "first_day": day - 1,
                       "mandatory_day": day, "recovery_attempts": 0,
                       "source_loc": target.get("source_loc") or self.loc,
                       "reason": "same_day_extra_did_not_complete_job"}
                self.deferred_weed_jobs.append(row)
                promoted.append(row)
                known.add(xy)
            if promoted:
                self.choice["weed_deferred"].append({
                    "day": day - 1, "loc": self.loc, "jobs": copy.deepcopy(promoted),
                    "detected_at_next_dawn": True,
                })
            self.pending_weed_jobs = []
        hashed = self.net.lookup(day, exact_hash(farm))
        aimed_id = None if not self.pending else int(self.pending.get("id") or -1)
        dest_farm = None
        dest_prev = None
        if self.pending is not None:
            dest_loc = str(self.pending.get("example") or self.loc or "")
            dest_farm = self.tapes.dawn_farm(dest_loc, day)
            if day > 0:
                dest_prev = self.tapes.dawn_farm(dest_loc, day - 1)
        if dest_farm is not None and self.deferred_weed_jobs:
            active = []
            for job in self.deferred_weed_jobs:
                x, y = (int(v) for v in job["xy"])
                live_tile = farm["tiles"][y][x]
                dest_tile = dest_farm["tiles"][y][x]
                if not is_weed(live_tile) and job.get("cleared_day") is None:
                    job["cleared_day"] = day
                    self.choice["weed_deferred_cleared"].append({
                        "day": day, "xy": [x, y], "first_day": job.get("first_day"),
                        "mandatory_day": job.get("mandatory_day"), "job": job.get("job"),
                    })
                if is_weed(live_tile) and day > int(job.get("mandatory_day") or day):
                    self.choice["weed_deferred_hard_failures"].append({
                        "day": day, "xy": [x, y], "reason": "weed_survived_mandatory_recovery_day",
                        "job": job.get("job"),
                    })
                if ((is_idle(live_tile) and is_idle(dest_tile))
                        or exact_cell(live_tile) == exact_cell(dest_tile)):
                    self.choice["weed_deferred_rejoined"].append({
                        "day": day, "xy": [x, y], "first_day": job.get("first_day"),
                        "job": job.get("job"),
                    })
                else:
                    active.append(job)
            self.deferred_weed_jobs = active
        if dest_farm is not None and self.deferred_cash_jobs:
            active = []
            for job in self.deferred_cash_jobs:
                x, y = (int(v) for v in job["xy"])
                live_tile = farm["tiles"][y][x]
                dest_tile = dest_farm["tiles"][y][x]
                if ((is_idle(live_tile) and is_idle(dest_tile))
                        or production_cell_match(live_tile, dest_tile)):
                    self._choice_log("cash_local_rejoined", {
                        "day": day, "xy": [x, y], "first_day": job["first_day"],
                        "mode": "source_tile_restored",
                    })
                else:
                    # Follow the current branch, not the old missed crop, if
                    # this tile has since changed its intended production.
                    if isinstance(dest_tile, dict) and dest_tile.get("kind") == "PLANT":
                        job["job"] = ["PLANT", dest_tile["crop"]]
                    active.append(job)
            self.deferred_cash_jobs = active
        prod_ok = bool(dest_farm is not None and prod_match(farm, dest_farm, dest_prev))
        cash_coords = {tuple(int(v) for v in row["xy"]) for row in self.deferred_cash_jobs}
        deferred_coords = cash_coords | {tuple(int(v) for v in row["xy"]) for row in self.deferred_weed_jobs}
        deferred_ok = bool(dest_farm is not None and deferred_coords
                           and _prod_match_except(farm, dest_farm, deferred_coords))
        exact_ok = hashed is not None and (aimed_id is None or int(hashed["id"]) == aimed_id)
        # Candidate index: production signature match, random idle weeds ignored.
        on_node = (prod_ok if dest_farm is not None else exact_ok) or deferred_ok or hashed is not None
        if prod_ok:
            self.choice["prod_hits"] = int(self.choice.get("prod_hits") or 0) + 1
        if prod_ok and not exact_ok:
            self.choice["exact_miss_prod_hit"] = int(self.choice.get("exact_miss_prod_hit") or 0) + 1
        if deferred_ok:
            if cash_coords:
                self._choice_log("cash_local_tolerated_dawns", {
                    "day": day, "aimed_id": aimed_id,
                    "coordinates": [list(xy) for xy in sorted(cash_coords)],
                })
            if self.deferred_weed_jobs:
                self.choice.setdefault("weed_deferred_tolerated_dawns", []).append({
                    "day": day, "aimed_id": aimed_id,
                    "coordinates": [list(xy) for xy in sorted(deferred_coords)],
                })
        if dest_farm is None and hashed is not None and aimed_id is None:
            on_node = True
        self.tape = []
        self.weed_plan = {}
        self.extra_hand_plan = None
        if on_node:
            self.hits += 1
            if (prod_ok or deferred_ok) and aimed_id is not None:
                self.node_id = int(aimed_id)
            elif hashed is not None:
                self.node_id = int(hashed["id"])
                if aimed_id is not None and self.node_id != aimed_id:
                    self.choice.setdefault("reattachments", []).append({
                        "day": day, "from": aimed_id, "to": self.node_id,
                    })
            self.choice["hits"] = self.hits
            self.pending = None
            edges = list(self.net.outgoing.get(self.node_id) or [])
            if edges:
                edge = None
                if self.preselected_edge is not None:
                    edge = next(
                        (candidate for candidate in edges if int(candidate["dst"]) == int(self.preselected_edge["dst"])),
                        None,
                    )
                self.preselected_edge = None
                if edge is None:
                    edge = self._select_edge(obs, edges)
                if edge is not None:
                    dest = self.net.dest_of(edge) or {}
                    loc = str(edge.get("example") or dest.get("example") or "")
                    self.loc = loc
                    self.pending = dest
                    self.choice["path"].append(
                        {
                            "day": day,
                            "src": int(self.node_id),
                            "dst": int(dest.get("id") or self.node_id),
                            "loc": loc,
                            "leaf": day == 29,
                        }
                    )
                    self.program_id = f"tape_{dest.get('id')}"
                else:
                    self.loc = ""
                    self.pending = None
            elif day == 29:
                # The graph ends at the day-29 dawn. The final 23 source
                # actions still exist, but use the reattached node's witness
                # when the originally aimed node was missed.
                if not prod_ok and hashed is not None:
                    self.loc = str(hashed.get("example") or "")
                if self.loc:
                    self.choice["path"].append({"day": day, "src": int(self.node_id), "dst": None, "loc": self.loc, "leaf": True})
            else:
                self.loc = ""
                self.pending = None
        else:
            self.misses += 1
            self.choice["misses"] = self.misses
            self.preselected_edge = None
            if self.choice.get("first_miss") is None:
                differences = []
                if dest_farm is not None:
                    for y, row in enumerate(dest_farm.get("tiles") or []):
                        for x, tile in enumerate(row or []):
                            live_tile = farm["tiles"][y][x]
                            if is_idle(tile) and is_idle(live_tile):
                                continue
                            if exact_cell(tile) != exact_cell(live_tile):
                                differences.append({"xy": [x, y], "expected": tile, "actual": live_tile})
                self.choice["first_miss"] = {
                    "day": day, "loc": self.loc, "aimed_id": aimed_id,
                    "actual_money": farm.get("money"),
                    "source_money": dest_farm.get("money") if dest_farm is not None else None,
                    "tile_differences": differences[:6], "tile_differences_count": len(differences),
                }
            # The stale tape is not a valid execution plan after a production
            # miss. Dropping it prevents a single failed PLANT/FEED from
            # silently contaminating every later day. Try reattachment again
            # from the live grid at the following dawn.
            self.choice.setdefault("off_graph_days", []).append(day)
            self.loc = ""
            self.pending = None
        if self.loc:
            self.tape = self.tapes.day_tape(self.loc, day)
            if self.market_mode == "derived":
                removed = remove_dead_animal_pickups(self.tape)
                if removed:
                    self.choice["dead_animal_pickups_removed"].append({
                        "day": day, "loc": self.loc, "actions": removed,
                    })
                trimmed = trim_consumable_pickups(self.tape)
                if trimmed:
                    self.choice["consumable_pickups_trimmed"].append({
                        "day": day, "loc": self.loc, "actions": trimmed,
                    })
            # The witness schedule is a timed sequence, not a daily shopping
            # total. In particular, SELL may fund later BUY/HIRE orders and a
            # dawn lump of purchases can exceed the ten-order turn limit.
            # Preserve every source market order at its original hour.
            self.shop_days += 1
            self.choice["shop_days"] = self.shop_days
            self.choice["tapes"] = int(self.choice.get("tapes") or 0) + 1
            contract = dawn_requirements(
                self.tape[:1] if self.market_mode in ("daily", "derived") else self.tape,
                rely_on_intraday_buys=self.market_mode not in ("daily", "derived"),
            )
            private = obs.get("private") or {}
            shortage = []
            for stock_key in ("seeds", "shed"):
                actual = private.get(stock_key) or {}
                for item, minimum in contract[stock_key].items():
                    if int(actual.get(item) or 0) < int(minimum):
                        shortage.append({"stock": stock_key, "item": item, "need": minimum, "have": int(actual.get(item) or 0)})
            if day > 0 and shortage:
                self.choice["resource_shortages"].append({"day": day, "loc": self.loc, "items": shortage})
            if self.repair_weeds and on_node and self.tape:
                if self.deferred_weed_jobs:
                    for job in self.deferred_weed_jobs:
                        job["recovery_attempts"] = int(job.get("recovery_attempts") or 0) + 1
                    recovered, recovery_remaining = repair_deferred_jobs(
                        farm, self.tape, self.deferred_weed_jobs,
                    )
                    if recovered:
                        self.choice["weed_deferred_recovery"].append({
                            "day": day, "loc": self.loc, "mode": "source_workers",
                            "jobs": recovered,
                        })
                    if recovery_remaining:
                        self.extra_hand_plan = plan_deferred_extra_hand(
                            farm, self.tape, recovery_remaining, len(farm.get("tiles") or []),
                        )
                        if self.extra_hand_plan is not None:
                            self.choice["weed_extra_planned"] += 1
                            self.choice["weed_deferred_recovery"].append({
                                "day": day, "loc": self.loc, "mode": "mandatory_extra_hand",
                                "jobs": recovery_remaining,
                                "hire_hour": self.extra_hand_plan["hire_hour"],
                                "actions": self.extra_hand_plan["actions"],
                            })
                            self.choice.setdefault("weed_extra_targets", []).append({
                                "day": day, "loc": self.loc,
                                "targets": self.extra_hand_plan["targets"],
                                "hire_hour": self.extra_hand_plan["hire_hour"],
                                "dig_hours": self.extra_hand_plan["dig_hours"],
                                "handoffs": {}, "mandatory_recovery": True,
                            })
                        else:
                            self.choice["weed_deferred_hard_failures"].append({
                                "day": day, "loc": self.loc, "reason": "no_mandatory_extra_route",
                                "jobs": recovery_remaining,
                            })
                if self.repair_cash and self.deferred_cash_jobs:
                    recovered, recovery_remaining = repair_deferred_jobs(
                        farm, self.tape, self.deferred_cash_jobs,
                    )
                    inserted = [row for row in recovered
                                if row.get("mode") != "already_present"]
                    if inserted:
                        self._choice_log("cash_local_recovery", {
                            "day": day, "loc": self.loc,
                            "mode": "source_workers", "jobs": inserted,
                        })
                    # An occupied lagging crop is local debt, not a reason to
                    # spend on an extra worker who cannot PLANT over it.
                    eligible = [row for row in recovery_remaining
                                if row.get("reason") != "deferred_tile_occupied"]
                    if eligible and self.extra_hand_plan is None:
                        extra = plan_deferred_extra_hand(
                            farm, self.tape, eligible, len(farm.get("tiles") or []),
                        )
                        if extra is not None:
                            self.extra_hand_plan = extra
                            self._choice_log("cash_local_recovery", {
                                "day": day, "loc": self.loc,
                                "mode": "extra_hand_planned", "jobs": eligible,
                                "hire_hour": extra["hire_hour"],
                            })
                    if recovery_remaining:
                        self._choice_log("cash_local_blocked", {
                            "day": day, "loc": self.loc, "jobs": recovery_remaining,
                        })
                self.weed_plan, blocked, arrival_repairs = repair_first_arrival_weeds(farm, self.tape)
                self.choice["weed_repairs_planned"] += len(self.weed_plan)
                shifted_modes = {"shift_into_pass", "deadline_worker_full_day_shift"}
                shifted = sum(row.get("mode") in shifted_modes for row in arrival_repairs)
                self.choice["weed_routes_changed"] += shifted
                self.choice["weed_delays"] += shifted
                self.choice["weed_first_arrival_repairs"] += len(arrival_repairs)
                if arrival_repairs:
                    self.choice["weed_first_arrival_details"].append({
                        "day": day, "loc": self.loc, "repairs": arrival_repairs,
                    })
                if shifted:
                    field_dependencies = repair_same_hour_field_dependencies(self.tape)
                    if field_dependencies:
                        self.choice["weed_dependency_delays"].append({
                            "day": day, "loc": self.loc, "actions": field_dependencies,
                        })
                    dependent = repair_same_hour_build_place(self.tape)
                    if dependent:
                        self.choice["weed_dependency_delays"].append({
                            "day": day, "loc": self.loc, "actions": dependent,
                        })
                    self.choice["weed_dependency_swaps"] += repair_same_hour_plant_water(self.tape)
                if blocked:
                    # Only hire after every source worker that reaches the
                    # tile, including the original worker at the deadline,
                    # failed to absorb the extra DIG safely.
                    same_day_extra = None if self.extra_hand_plan is not None else plan_extra_hand_targets(
                        self.tape, blocked, len(farm.get("tiles") or []),
                    )
                    if same_day_extra is not None:
                        self.extra_hand_plan = same_day_extra
                        self.pending_weed_jobs = [
                            {**copy.deepcopy(target), "source_loc": self.loc} for target in blocked
                        ]
                        self.choice["weed_extra_planned"] += 1
                        self.choice.setdefault("weed_extra_targets", []).append({
                            "day": day, "loc": self.loc,
                            "targets": self.extra_hand_plan["targets"],
                            "hire_hour": self.extra_hand_plan["hire_hour"],
                            "dig_hours": self.extra_hand_plan["dig_hours"],
                            "handoffs": self.extra_hand_plan.get("handoffs") or {},
                        })
                        blocked = []
                if blocked:
                    self.choice["weed_uncovered"].append({"day": day, "loc": self.loc, "targets": blocked[:20]})
                    known = {tuple(int(v) for v in row["xy"]) for row in self.deferred_weed_jobs}
                    deferred_now = []
                    for target in blocked:
                        xy = tuple(int(v) for v in target["xy"])
                        if xy in known:
                            continue
                        job = list(target.get("job") or [])
                        if not job:
                            deadline, worker = int(target["deadline_hour"]), int(target["worker"])
                            units = [self.tape[deadline].get("farmer") or ["PASS"],
                                     *(self.tape[deadline].get("hands") or [])]
                            job = list(units[worker])
                        row = {"xy": list(xy), "job": job, "first_day": day,
                               "mandatory_day": day + 1, "recovery_attempts": 0,
                               "source_loc": self.loc}
                        self.deferred_weed_jobs.append(row)
                        deferred_now.append(row)
                        known.add(xy)
                    if deferred_now:
                        self.choice["weed_deferred"].append({
                            "day": day, "loc": self.loc, "jobs": copy.deepcopy(deferred_now),
                        })

    def _play(self, obs, hour: int) -> dict:
        if self.repair_cash:
            repaired, blocked = postpone_unfunded_plants(
                own_farm(obs), obs.get("private") or {}, self.tape, hour,
                protected_slots=set(self.weed_plan),
            )
            for row in repaired:
                self._choice_log("cash_seed_delays", {"day": int(obs.get("day") or 0), **row})
            for row in blocked:
                self._choice_log("cash_seed_unresolved", {"day": int(obs.get("day") or 0), **row})
                xy = tuple(int(v) for v in row["xy"])
                farm = own_farm(obs)
                if not is_idle(farm["tiles"][xy[1]][xy[0]]):
                    continue
                if xy not in {tuple(int(v) for v in job["xy"])
                              for job in self.deferred_cash_jobs}:
                    job = {"xy": list(xy), "job": ["PLANT", row["crop"]],
                           "first_day": int(obs.get("day") or 0),
                           "source_loc": self.loc, "reason": "unfunded_seed"}
                    self.deferred_cash_jobs.append(job)
                    self._choice_log("cash_local_deferred", job)
        if self.market_mode == "derived" and not self.extra_hand_plan and not self.weed_plan:
            deposit = plan_early_deposit(obs, self.tape, hour)
            if deposit:
                self.choice["early_deposits"].append(deposit)
        step = self.tape[hour] if hour < len(self.tape) else {"farmer": ["PASS"], "hands": [], "hires": 0}
        farmer = list(step.get("farmer") or ["PASS"])
        hands = []
        for hand in step.get("hands") or []:
            hands.append(list(hand) if isinstance(hand, list) and hand else ["PASS"])
        if self.weed_plan:
            farmer, hands, used = apply_idle_digs(own_farm(obs), hour, farmer, hands, self.weed_plan)
            if used:
                self.choice["weed_repairs_used"] += 1
        source_market = [] if self.market_mode == "derived" else [
            list(order) for order in (step.get("market") or []) if isinstance(order, list) and order
        ]
        if self.market_mode == "raw_source":
            return {"farmer": farmer, "hands": hands, "market": source_market}
        if source_market:
            source_market = _cap_structural_orders(source_market, step, own_farm(obs), hour)
        if self.market_mode in ("inverse", "planned"):
            # The worker tape is the target. Replay HIRE/LAND at their source
            # times, but derive all inputs from actual stock and next-hour
            # unit demand; historical BUY/SELL churn is not a contract.
            allowed = ("HIRE", "BUY_LAND") if self.market_mode == "inverse" else (
                "HIRE", "BUY_LAND", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"
            )
            source_market = [order for order in source_market if order[0] in allowed]
        if self.extra_hand_plan is not None:
            hire_hour = int(self.extra_hand_plan["hire_hour"])
            if hour == hire_hour:
                mandatory_recovery = bool(self.extra_hand_plan.get("mandatory_recovery"))
                affordability_orders = list(source_market)
                input_unmet = []
                if self.market_mode == "derived":
                    live = own_farm(obs)
                    target_hands = step.get("post_hands")
                    target_land = step.get("post_land")
                    affordability_orders.extend(
                        [["HIRE"] for _ in range(max(0, int(target_hands) - len(live.get("hands") or [])))]
                        if target_hands is not None and hour < 23 else []
                    )
                    affordability_orders.extend(
                        [["BUY_LAND"] for _ in range(max(0, int(target_land) - len(live.get("unlocked_quadrants") or [])))]
                        if target_land is not None else []
                    )
                    if not mandatory_recovery:
                        # The weed hand is a current hard obligation.  Do not
                        # let speculative/full-day prebuy consume its order
                        # slot or cash.  Only stock required by the very next
                        # field phase may precede it; later needs remain in the
                        # persistent market backlog and are issued over time.
                        projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
                        future = self.tape[hour + 1:hour + 2]
                        needed = dawn_requirements(advance_purchase_horizon(future, projected), rely_on_intraday_buys=False)
                        buys, input_unmet = top_up_orders(
                            needed, projected, max_orders=max(0, 9 - len(affordability_orders)),
                        )
                        affordability_orders.extend(buys)
                affordable, remaining, cost = _extra_hire_affordable(
                    obs, affordability_orders, reserve=0,
                )
                prefix_ok = len(own_farm(obs).get("hands") or []) == len(step.get("hands") or [])
                slot_ok = len(affordability_orders) < 10
                input_ok = mandatory_recovery or not input_unmet
                # Derived mode requests the recovery hire even when cash is
                # currently short. The scheduler may defer it; a trailing
                # sale can only fund a retry in the following market phase.
                liquidity_managed = self.market_mode == "derived"
                if (affordable or liquidity_managed) and input_ok and slot_ok and prefix_ok:
                    source_market.append(["HIRE"])
                    self.extra_hand_plan["requested"] = True
                    self.choice["weed_extra_requested"] += 1
                    if liquidity_managed and not affordable:
                        self._choice_log("weed_extra_liquidity_obligations", {
                            "day": int(obs.get("day") or 0), "hour": hour,
                            "remaining_before_sale": round(remaining, 2),
                            "extra_cost": round(cost, 2),
                        })
                else:
                    reasons = []
                    if not affordable and not liquidity_managed:
                        reasons.append("current_cash")
                    if not input_ok:
                        reasons.append("next_hour_input")
                    if not slot_ok:
                        reasons.append("market_order_limit")
                    if not prefix_ok:
                        reasons.append("source_worker_prefix")
                    self.choice["weed_extra_skipped"].append({
                        "day": int(obs.get("day") or 0), "hour": hour,
                        "remaining_after_source": round(remaining, 2),
                        "extra_cost": round(cost, 2), "order_slots": 10 - len(source_market),
                        "input_unmet": input_unmet, "mandatory_recovery": mandatory_recovery,
                        "reasons": reasons,
                    })
            elif hour > hire_hour and self.extra_hand_plan.get("requested"):
                actual_hands = own_farm(obs).get("hands") or []
                if len(actual_hands) == len(hands) + 1:
                    bonus = list(self.extra_hand_plan["actions"].get(hour, ["PASS"]))
                    target = self.extra_hand_plan.get("dig_targets", {}).get(hour, self.extra_hand_plan["xy"])
                    if bonus == ["DIG"] and list(actual_hands[-1]) != list(target):
                        bonus = ["PASS"]
                    handoff = (self.extra_hand_plan.get("handoffs") or {}).get(hour)
                    if handoff:
                        farm = own_farm(obs)
                        positions = [farm.get("farmer"), *actual_hands]
                        worker = int(handoff["worker"])
                        x, y = handoff["xy"]
                        source_action = farmer if worker == 0 else hands[worker - 1]
                        if (positions[worker] == [x, y] and positions[-1] == [x, y]
                                and source_action == handoff["job"] and is_weed(farm["tiles"][y][x])):
                            if worker == 0:
                                farmer = ["DIG"]
                            else:
                                hands[worker - 1] = ["DIG"]
                            self.choice["weed_extra_handoffs"].append({
                                "day": int(obs.get("day") or 0), "hour": hour, **handoff,
                            })
                        else:
                            bonus = ["PASS"]
                    hands.append(bonus)
        if self.repair_cash and any(
            int(row.get("day", -1)) == int(obs.get("day") or 0)
            for row in self.choice.get("market_order_shortages") or []
        ):
            farmer, hands, handoffs = handoff_missed_maintenance(
                own_farm(obs), obs.get("private") or {}, step, farmer, hands,
            )
            for row in handoffs:
                self._choice_log("cash_maintenance_handoffs", {
                    "day": int(obs.get("day") or 0), "hour": hour, **row,
                })
        if self.market_mode == "daily":
            if hour >= 20 and int(obs.get("day") or 0) < 29:
                next_edge = self._preselect_next_edge(obs)
                next_tape = self.tapes.day_tape(str(next_edge.get("example") or ""), int(obs.get("day") or 0) + 1) if next_edge else []
                # The final four hours are the next day's procurement window.
                # Include only its first four unit phases, not a multi-day
                # stockpile, while still allowing source advance-buy timing.
                future_tape = self.tape[hour + 1 :] + next_tape[:4]
            else:
                future_tape = self.tape[hour + 1 :]
            projected_now = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
            source_market = _cap_daily_buys(source_market, projected_now, future_tape)
        if self.market_mode == "derived":
            return self._play_derived_market(obs, hour, step, farmer, hands, source_market)
        if self.market_mode in ("inverse", "planned"):
            return self._play_inverse_market(obs, hour, farmer, hands, source_market)
        day = int(obs.get("day") or 0)
        next_contract = None
        if hour == 23 and day < 29 and self.pending is not None:
            # Day d+1 can require PLANT/PICKUP before its first market phase.
            # Select its edge on day d's last actionable hour, then buy only
            # the dawn-prefix shortfall for that selected worker schedule.
            next_edge = self._preselect_next_edge(obs)
            if next_edge is not None:
                future_loc = str(next_edge.get("example") or "")
                future_tape = self.tapes.day_tape(future_loc, day + 1)
                contract = dawn_requirements(
                    future_tape[:1] if self.market_mode == "daily" else future_tape,
                    rely_on_intraday_buys=self.market_mode != "daily",
                )
                next_contract = contract
                projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
                for order in source_market:
                    if order[0] == "SELL" and len(order) >= 3:
                        item = str(order[1])
                        projected["shed"][item] = max(0, int(projected["shed"].get(item) or 0) - int(order[2] or 0))
                    elif order[0] == "BUY_SEED" and len(order) >= 3:
                        item = str(order[1])
                        projected["seeds"][item] = int(projected["seeds"].get(item) or 0) + int(order[2] or 0)
                    elif order[0] in ("BUY_ANIMAL", "BUY_PRODUCT") and len(order) >= 3:
                        item = str(order[1])
                        projected["shed"][item] = int(projected["shed"].get(item) or 0) + int(order[2] or 0)
                prep_orders, unmet = top_up_orders(contract, projected, max_orders=max(0, 10 - len(source_market)))
                self.choice["resource_prep"].append({
                    "day": day, "for_day": day + 1, "dst": int(next_edge["dst"]),
                    "orders": prep_orders, "unmet": unmet, "max_active_hands": contract["max_active_hands"],
                })
            else:
                prep_orders = []
        else:
            prep_orders = []
        if self.market_mode == "daily" and hour < 23:
            # Market purchases at h can support the h+1 field phase, not h.
            # The full-day cap above prevents buying for a later day.
            next_contract = dawn_requirements(self.tape[hour + 1 : hour + 2], rely_on_intraday_buys=False)
            projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
            for order in source_market:
                if len(order) < 3:
                    continue
                item = str(order[1])
                qty = int(order[2] or 0)
                if order[0] == "SELL":
                    projected["shed"][item] = max(0, int(projected["shed"].get(item) or 0) - qty)
                elif order[0] == "BUY_SEED":
                    projected["seeds"][item] = int(projected["seeds"].get(item) or 0) + qty
                elif order[0] in ("BUY_ANIMAL", "BUY_PRODUCT"):
                    projected["shed"][item] = int(projected["shed"].get(item) or 0) + qty
            prep_orders, unmet = top_up_orders(next_contract, projected, max_orders=max(0, 10 - len(source_market)))
            if unmet:
                self.choice["resource_shortages"].append({
                    "day": day, "hour": hour, "loc": self.loc, "unmet": unmet,
                })
        if self.sell_mode == "source":
            market = source_market + prep_orders
        else:
            turn = int(obs.get("day") or 0) * 24 + hour
            if turn == 718:
                # Last legal action: units act before the market. Include any
                # produce deposited by a DROP on this same turn, then liquidate
                # every sellable shed item. No future production order remains.
                farmer, hands = _terminal_drop_idle(obs, farmer, hands)
                available = _post_unit_shed(obs, {"farmer": farmer, "hands": hands})
                market = sell_immediate(available, {}) if self.sell_mode == "immediate" else sell(obs, available, {})
                sold = {str(order[1]): int(order[2]) for order in market}
                self.choice["terminal_unsold_projected"] = {
                    item: int(available.get(item) or 0) - int(sold.get(item) or 0)
                    for item in SELLABLE if int(available.get(item) or 0) > int(sold.get(item) or 0)
                }
                projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
                self.choice["terminal_unsold_bags_projected"] = {
                    item: sum(int((bag or {}).get(item) or 0) for bag in (projected.get("inventories") or []))
                    for item in SELLABLE
                    if any(int((bag or {}).get(item) or 0) > 0 for bag in (projected.get("inventories") or []))
                }
            else:
                if len(source_market) > 10:
                    raise ValueError(f"source has {len(source_market)} market orders at hour {hour}")
                available = _post_unit_shed(obs, {"farmer": farmer, "hands": hands})
                # Reserve the next required *prefix* stock, not the full-day
                # pickup sum: wheat bought or deposited before later pickups
                # does not have to sit in the shed now. This frees surplus to
                # fund same-turn HIRE/seed orders without starving the herd.
                later_contract = dawn_requirements(
                    self.tape[hour + 1 :], rely_on_intraday_buys=True,
                    include_source_sales=True,
                )
                reserved = {
                    item: int((later_contract.get("shed") or {}).get(item) or 0)
                    for item in ("WHEAT", "FERTILIZER")
                }
                if next_contract is not None:
                    for item in reserved:
                        reserved[item] = max(reserved[item], int((next_contract.get("shed") or {}).get(item) or 0))
                elif self.market_mode == "daily" and hour >= 20 and day < 29 and self.preselected_edge is not None:
                    future_loc = str(self.preselected_edge.get("example") or "")
                    future_tape = self.tapes.day_tape(future_loc, day + 1)
                    early_need = dawn_requirements(future_tape[:4], rely_on_intraday_buys=False)
                    for item in reserved:
                        reserved[item] = max(reserved[item], int((early_need.get("shed") or {}).get(item) or 0))
                sales = sell_immediate(available, reserved) if self.sell_mode == "immediate" else sell(obs, available, reserved)
                # Keep the source's timed sales as a liquidity floor. Replacing
                # them outright can make a later HIRE/BUY fail in the same turn.
                # The established seller may liquidate additional surplus only
                # after those source obligations and only in free order slots.
                already_sold = {str(order[1]) for order in source_market if order[0] == "SELL" and len(order) >= 3}
                extra_sales = [order for order in sales if str(order[1]) not in already_sold]
                free_for_sales = max(0, 10 - len(source_market) - len(prep_orders))
                # Extra sales precede the source's purchase/hire block so their
                # proceeds can fund it in this same market phase. Source order
                # relative order is otherwise unchanged.
                market = extra_sales[:free_for_sales] + source_market + prep_orders
        if len(market) > 10:
            raise ValueError(f"market has {len(market)} orders at hour {hour}")
        return {"farmer": farmer, "hands": hands, "market": market}

    def _play_derived_market(self, obs, hour: int, step: dict, farmer: list, hands: list[list],
                             extra_orders: list[list]) -> dict:
        """Use source field instructions only; derive every market order live.

        At market hour h, workers have already acted. This market can provision
        hour h+1. The h23 market is the last chance to provision tomorrow h0.
        """
        day = int(obs.get("day") or 0)
        if day * 24 + hour == 718:
            farmer, hands = _terminal_drop_idle(obs, farmer, hands)
            projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
            if self.sell_mode == "seller":
                plan = self.market_seller.plan(
                    obs, projected.get("shed") or {}, {}, terminal=True,
                    available_order_slots=10,
                )
                market = plan.orders
                self._choice_log("market_sell_plans", {
                    "day": day, "hour": hour, "terminal": True,
                    "orders": market, "expected_revenue": round(plan.expected_revenue, 3),
                })
                accounting = self.choice["market_sell_accounting"]
                accounting["planned_orders"] += len(market)
                accounting["issued_orders"] += len(market)
                terminal_units = sum(int(order[2] or 0) for order in market)
                accounting["planned_units"] += terminal_units
                accounting["issued_units"] += terminal_units
                terminal_estimate = self.market_seller.stressed_revenue(obs, market)
                accounting["planned_stressed_revenue_estimate"] += terminal_estimate
                accounting["issued_stressed_revenue_estimate"] += terminal_estimate
            else:
                market = sell_immediate(projected.get("shed") or {}, {})[:10]
            self.choice["terminal_market_backlog"] = [
                {k: v for k, v in row.items() if k != "last_submitted_turn"}
                for row in sorted(self.market_backlog.values(), key=lambda value: int(value.get("seq") or 0))
                if int(row.get("remaining") or 0) > 0
            ]
            self.choice["terminal_unsold_projected"] = {
                item: int(n or 0) for item, n in (projected.get("shed") or {}).items()
                if item in SELLABLE and int(n or 0) > 0 and item not in {str(o[1]) for o in market}
            }
            return {"farmer": farmer, "hands": hands, "market": market}

        farm = own_farm(obs)
        projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
        current_future = self.tape[hour + 1 :]
        next_tape = []
        if hour == 23 and day < 29:
            edge = self._preselect_next_edge(obs)
            next_tape = self.tapes.day_tape(str(edge.get("example") or ""), day + 1) if edge else []
            if next_tape:
                remove_dead_animal_pickups(next_tape)
                trim_consumable_pickups(next_tape)
            future = next_tape
            after_reset = copy.deepcopy(projected)
            engine._drop_inventories_to_shed(after_reset, 100)
            stock_for_topup = after_reset
        else:
            edge = None
            future = current_future
            stock_for_topup = projected
        purchase_horizon = advance_purchase_horizon(future, stock_for_topup)
        urgent = dawn_requirements(purchase_horizon, rely_on_intraday_buys=False)
        immediate = dawn_requirements(future[:1], rely_on_intraday_buys=False)
        remaining = dawn_requirements(current_future, rely_on_intraday_buys=False)
        extra_seed_priorities = {}
        if self.extra_hand_plan is not None and self.extra_hand_plan.get("requested"):
            # A deferred PLANT is not present in the source tape's unit events.
            # Reserve its seed explicitly so atomic PLANT validation cannot
            # cancel both the recovery job and ordinary source planting.
            extra_seed_need = {}
            for action_hour, action in (self.extra_hand_plan.get("actions") or {}).items():
                if int(action_hour) > hour and action and action[0] == "PLANT" and len(action) >= 2:
                    crop = str(action[1])
                    extra_seed_need[crop] = int(extra_seed_need.get(crop) or 0) + 1
                    offset = int(action_hour) - (int(hour) + 1)
                    extra_seed_priorities[crop] = min(
                        int(extra_seed_priorities.get(crop, offset)), offset,
                    )
            for crop, amount in extra_seed_need.items():
                urgent.setdefault("seeds", {})[crop] = int((urgent.get("seeds") or {}).get(crop) or 0) + amount
                remaining.setdefault("seeds", {})[crop] = int((remaining.get("seeds") or {}).get(crop) or 0) + amount
                if int(extra_seed_priorities.get(crop, 99)) == 0:
                    immediate.setdefault("seeds", {})[crop] = int((immediate.get("seeds") or {}).get(crop) or 0) + amount
        reserved = dict(remaining.get("shed") or {})
        for item, amount in (urgent.get("shed") or {}).items():
            reserved[item] = max(int(reserved.get(item) or 0), int(amount))

        board_size = len(farm.get("tiles") or []) or 10
        self._ensure_hire_plan(day, board_size)
        planned_hire_hours = [int(value) for value in (self.hire_plan.get("hours") or [])]
        source_before = sum(value < hour for value in planned_hire_hours)
        source_after = sum(value <= hour for value in planned_hire_hours)
        actual_source_prefix = min(len(farm.get("hands") or []), source_before)
        hire_count = max(0, source_after - actual_source_prefix) if hour < 23 else 0
        target_land = step.get("post_land")
        land_count = max(0, int(target_land) - len(farm.get("unlocked_quadrants") or [])) if target_land is not None else 0
        staffing = [["HIRE"] for _ in range(hire_count)] + [["BUY_LAND"] for _ in range(land_count)]
        # The supplementary weed worker is requested after source workers, so
        # its index cannot shift the witness instructions.
        staffing.extend([list(o) for o in extra_orders if o and o[0] == "HIRE"])
        turn = day * 24 + hour
        priorities = first_stock_use_offsets(purchase_horizon)
        for crop, priority in extra_seed_priorities.items():
            key = ("BUY_SEED", str(crop))
            priorities[key] = min(int(priorities.get(key, priority)), int(priority))
        requested, unmet = contract_order_gaps(urgent, stock_for_topup)
        immediate_requested, _immediate_unmet = contract_order_gaps(immediate, stock_for_topup)
        immediate_gaps = {
            (str(op), str(item)): max(0, int(quantity or 0))
            for op, item, quantity in immediate_requested
        }
        # ``advance_purchase_horizon`` expands beyond one field phase only
        # when the following market would otherwise have too few order slots
        # after its fixed hire/land prefix.  Those advanced inputs are therefore
        # current execution obligations, not optional prebuys: postponing one
        # recreates the exact eleven-orders-for-ten-slots failure the horizon
        # expansion was designed to prevent.
        slot_relief_contract = len(purchase_horizon) > 1
        for op, item, quantity in requested:
            key = (str(op), str(item))
            priority = max(0, int(priorities.get(key, 0 if key in immediate_gaps else 99)))
            self._queue_market_order(
                op, item, quantity, due_turn=turn + priority,
                priority=priority,
                reason=("slot_relief_contract" if slot_relief_contract
                        else "next_field_contract"),
                planned_turn=turn,
                hard_quantity=(
                    int(quantity) if slot_relief_contract
                    else min(int(quantity), int(immediate_gaps.get(key, 0)))
                ),
            )
        if unmet:
            self.choice["resource_shortages"].append({"day": day, "hour": hour, "loc": self.loc, "unmet": unmet})
        hard_buys, future_buys, stale_buys = self._market_backlog_groups(turn)
        future_staffing = self._future_source_staffing_orders(hour, farm, staffing)
        hard_bundle = [list(order) for order in hard_buys + staffing]
        hard_cost = _estimate_market_cost(obs, hard_bundle)
        hard_purchase_cost = _estimate_market_cost(obs, hard_buys)
        known_purchase_cost = _estimate_market_cost(obs, hard_buys + future_buys)
        finance_contract = dawn_requirements(
            future[:4], rely_on_intraday_buys=False,
        )
        finance_orders, _finance_unmet = contract_order_gaps(finance_contract, stock_for_topup)
        near_purchase_cost = _estimate_market_cost(obs, finance_orders)
        purchase_cash_total = max(known_purchase_cost, near_purchase_cost)
        future_purchase_reserve = max(0.0, purchase_cash_total - hard_purchase_cost)
        known_future_purchase_cost = max(
            0.0, known_purchase_cost - hard_purchase_cost,
        )
        unmaterialized_purchase_reserve = max(
            0.0, future_purchase_reserve - known_future_purchase_cost,
        )
        future_staffing_reserve = max(
            0.0,
            _estimate_market_cost(obs, staffing + future_staffing)
            - _estimate_market_cost(obs, staffing),
        )
        dawn_staffing_reserve = (
            next_day_staffing_reserve(next_tape, farm)
            if hour == 23 and next_tape else 0.0
        )
        staffing_cash_reserve = future_staffing_reserve + dawn_staffing_reserve
        protected_cash_after_hard = staffing_cash_reserve + unmaterialized_purchase_reserve
        capacity_need = sum(
            max(0, int(order[2] or 0))
            for order in hard_bundle
            if len(order) >= 3 and order[0] in ("BUY_ANIMAL", "BUY_PRODUCT")
        )
        # The h23 market runs before the engine's automatic end-of-day bag
        # deposit.  Holding shed stock through this phase can therefore discard
        # already-produced goods even though a legal sale could have made room.
        # This is a read-only capacity obligation for the seller; no worker or
        # procurement schedule is changed.
        day_end_drop_need = 0
        next_locked_market_capacity_need = 0
        if hour == 23:
            day_end_drop_need = sum(
                max(0, int(quantity or 0))
                for bag in (projected.get("inventories") or [])
                for quantity in (bag or {}).values()
            )
            if next_tape:
                # If tomorrow h0 is exactly filled by source hires/land and
                # required purchases, it has no SELL slot with which to make
                # room for a shed-bound purchase.  Reserve that room tonight;
                # this changes neither the purchase nor the worker schedule.
                next_hire_plan = plan_source_hire_schedule(next_tape, board_size)
                next_hires = sum(
                    int(hire_hour) == 0
                    for hire_hour in (next_hire_plan.get("hours") or [])
                )
                land_after_current = (
                    len(farm.get("unlocked_quadrants") or [])
                    + sum(bool(order and order[0] == "BUY_LAND") for order in staffing)
                )
                next_land_target = next_tape[0].get("post_land") if next_tape else None
                next_lands = (
                    max(0, int(next_land_target) - land_after_current)
                    if next_land_target is not None else 0
                )
                planned_dawn = copy.deepcopy(after_reset)
                for order in hard_buys:
                    if len(order) < 3:
                        continue
                    op, item, quantity = str(order[0]), str(order[1]), max(0, int(order[2] or 0))
                    if op == "BUY_SEED":
                        seeds = planned_dawn.setdefault("seeds", {})
                        seeds[item] = int(seeds.get(item) or 0) + quantity
                    elif op in ("BUY_ANIMAL", "BUY_PRODUCT"):
                        shed = planned_dawn.setdefault("shed", {})
                        shed[item] = int(shed.get(item) or 0) + quantity
                next_market_contract = dawn_requirements(
                    next_tape[:2], rely_on_intraday_buys=False,
                )
                next_market_orders, _ = contract_order_gaps(
                    next_market_contract, planned_dawn,
                )
                if next_hires + next_lands + len(next_market_orders) >= 10:
                    next_locked_market_capacity_need = sum(
                        max(0, int(order[2] or 0))
                        for order in next_market_orders
                        if len(order) >= 3 and order[0] in ("BUY_ANIMAL", "BUY_PRODUCT")
                    )
        if self.sell_mode == "seller":
            sell_plan = self.market_seller.plan(
                obs,
                projected.get("shed") or {},
                reserved,
                minimum_cash=hard_cost + future_purchase_reserve + staffing_cash_reserve,
                capital_floor=self.market_capital_floor,
                required_capacity=(capacity_need + day_end_drop_need
                                   + next_locked_market_capacity_need),
                # A full hard bundle leaves no same-turn sale slot. Liquidity
                # must be raised earlier or the missed obligation recorded.
                # The hard farm orders reserve slots first. Optional prebuys
                # can claim further slots; the scheduler then takes only the
                # highest-ranked sale signals that still fit.
                available_order_slots=max(0, 10 - len(hard_bundle)),
                projected_arrivals=_projected_sellable_arrivals(current_future),
            )
            all_sales = sell_plan.orders
            sell_log = {
                "day": day, "hour": hour,
                "orders": all_sales,
                "expected_revenue": round(sell_plan.expected_revenue, 3),
                "forced": sell_plan.forced,
                "funding_target": round(sell_plan.funding_target, 3),
                "funding_gap": round(sell_plan.funding_gap, 3),
                "funding_shortfall": round(sell_plan.funding_shortfall, 3),
                "capacity_release": sell_plan.capacity_release,
                "day_end_drop_need": day_end_drop_need,
                "next_locked_market_capacity_need": next_locked_market_capacity_need,
                "future_purchase_reserve": round(future_purchase_reserve, 3),
                "unmaterialized_purchase_reserve": round(unmaterialized_purchase_reserve, 3),
                "next_day_staffing_reserve": round(dawn_staffing_reserve, 3),
                "timing_quantities": sell_plan.timing_quantities,
                "forced_quantities": sell_plan.forced_quantities,
                "timing_details": sell_plan.timing_details,
            }
            if len(self.choice["market_sell_plans"]) < 256:
                self.choice["market_sell_plans"].append(sell_log)
            else:
                sell_log = None
        else:
            sell_log = None
            all_sales = sorted(
                sell_immediate(projected.get("shed") or {}, reserved),
                key=lambda order: (_estimate_sale_value(obs, [order]), int(order[2] or 0)),
                reverse=True,
            )
        market, issued_buys, issued_staffing = self._schedule_current_first_market(
            obs, projected, hard_buys, staffing, future_buys, stale_buys,
            all_sales, future_staffing, turn=turn, day=day, hour=hour,
            protected_cash_after_hard=protected_cash_after_hard,
        )
        if self.sell_mode == "seller":
            actual_sale_orders = [list(order) for order in market if order and order[0] == "SELL"]
            issued_estimate = self.market_seller.stressed_revenue(obs, actual_sale_orders)
            if sell_log is not None:
                sell_log["actual_orders"] = actual_sale_orders
                sell_log["omitted_orders"] = [list(order) for order in all_sales[len(actual_sale_orders):]]
                sell_log["actual_stressed_revenue_estimate"] = round(issued_estimate, 3)
            accounting = self.choice["market_sell_accounting"]
            accounting["planned_orders"] += len(all_sales)
            accounting["issued_orders"] += len(actual_sale_orders)
            accounting["planned_units"] += sum(int(order[2] or 0) for order in all_sales)
            accounting["issued_units"] += sum(int(order[2] or 0) for order in actual_sale_orders)
            accounting["planned_stressed_revenue_estimate"] += self.market_seller.stressed_revenue(obs, all_sales)
            accounting["issued_stressed_revenue_estimate"] += issued_estimate
            actual_sales: dict[str, int] = {}
            for order in market:
                if order and order[0] == "SELL" and len(order) >= 3:
                    item = str(order[1])
                    actual_sales[item] = actual_sales.get(item, 0) + int(order[2] or 0)
            self.market_seller.own_last_sales = actual_sales
        if (self.extra_hand_plan is not None and self.extra_hand_plan.get("requested")
                and any(order and order[0] == "HIRE" for order in extra_orders)
                and len(issued_staffing) < len(staffing)):
            self._defer_extra_hire(day, hour, "scheduler_hire_deferred")
        if len(issued_staffing) < len(staffing):
            staffing_deferred = [row for row in self.last_market_hard_deferred
                                 if row["order"][0] in ("HIRE", "BUY_LAND")]
            shortage_reason = (staffing_deferred[0]["reason"] if staffing_deferred
                               else "market_order_limit")
            self.choice["resource_shortages"].append({"day": day, "hour": hour, "loc": self.loc,
                                                      "unmet": [{"reason": f"staff_{shortage_reason}", "quantity": len(staffing) - len(issued_staffing)}]})
            self._choice_log("hire_schedule_shortages", {
                "day": day, "hour": hour, "requested": len(staffing),
                "issued": len(issued_staffing), "reason": shortage_reason,
            })
        if hour == 23 and day < 29:
            self.choice["resource_prep"].append({
                "day": day, "for_day": day + 1, "dst": int(edge["dst"]) if edge else None,
                "orders": issued_buys, "unmet": unmet, "max_active_hands": urgent["max_active_hands"],
                "backlog": [
                    {k: v for k, v in row.items() if k not in ("last_submitted_turn", "last_submitted_quantity")}
                    for row in sorted(self.market_backlog.values(), key=lambda value: int(value.get("seq") or 0))
                    if int(row.get("remaining") or 0) > 0
                ],
            })
        return {"farmer": farmer, "hands": hands, "market": market}

    def _play_inverse_market(self, obs, hour: int, farmer: list, hands: list[list], core_market: list[list]) -> dict:
        day = int(obs.get("day") or 0)
        turn = day * 24 + hour
        if turn == 718:
            farmer, hands = _terminal_drop_idle(obs, farmer, hands)
            available = _post_unit_shed(obs, {"farmer": farmer, "hands": hands})
            market = sell_immediate(available, {})
            projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
            self.choice["terminal_unsold_projected"] = {}
            self.choice["terminal_unsold_bags_projected"] = {
                item: sum(int((bag or {}).get(item) or 0) for bag in (projected.get("inventories") or []))
                for item in SELLABLE
                if any(int((bag or {}).get(item) or 0) > 0 for bag in (projected.get("inventories") or []))
            }
            return {"farmer": farmer, "hands": hands, "market": market}

        projected = _post_unit_private(obs, {"farmer": farmer, "hands": hands})
        if hour == 23 and day < 29:
            edge = self._preselect_next_edge(obs)
            future = self.tapes.day_tape(str(edge.get("example") or ""), day + 1) if edge else []
            next_day = day + 1 if edge else None
        else:
            future = self.tape[hour + 1 :]
            next_day = None
        # The market phase occurs after this hour's field actions. Buy the
        # stock needed *before* the following hour's field actions. If the
        # following hour is a new day, this is the selected successor tape.
        contract = dawn_requirements(future[:1], rely_on_intraday_buys=False)
        planning_horizon = future[:2] if next_day is not None else future
        reserve = dawn_requirements(planning_horizon, rely_on_intraday_buys=False)
        reserved = {"WHEAT": int((reserve.get("shed") or {}).get("WHEAT") or 0)}
        if next_day is not None and future:
            self.choice["resource_prep"].append({
                "day": day, "for_day": next_day,
                "dst": int(edge["dst"]), "contract": contract,
            })
        if self.market_mode == "planned":
            # Keep historical advance-buy timing, but only to the maximum
            # future unit demand on this selected tape. Source replays often
            # contain large speculative BUYs that did not contribute to farm
            # production, especially near the terminal step.
            capped = []
            already = copy.deepcopy(projected)
            future_need = dawn_requirements(planning_horizon, rely_on_intraday_buys=False)
            for order in core_market:
                op = str(order[0])
                if op not in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") or len(order) < 3:
                    capped.append(order)
                    continue
                item = str(order[1])
                stock_key = "seeds" if op == "BUY_SEED" else "shed"
                need = int((future_need.get(stock_key) or {}).get(item) or 0)
                have = int((already.get(stock_key) or {}).get(item) or 0)
                qty = min(int(order[2] or 0), max(0, need - have))
                if qty > 0:
                    capped.append([op, item, qty])
                    already[stock_key][item] = have + qty
            core_market = capped
        after_core = copy.deepcopy(projected)
        for order in core_market:
            if len(order) < 3 or order[0] not in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"):
                continue
            stock_key = "seeds" if order[0] == "BUY_SEED" else "shed"
            item = str(order[1])
            after_core[stock_key][item] = int(after_core[stock_key].get(item) or 0) + int(order[2] or 0)
        prelim_buys, _ = top_up_orders(contract, after_core, max_orders=max(0, 10 - len(core_market)))
        sales = sell_immediate(projected.get("shed") or {}, reserved)
        sale_slots = max(0, 10 - len(core_market) - len(prelim_buys))
        sales = sales[:sale_slots]
        after_sales = copy.deepcopy(after_core)
        for order in sales:
            item = str(order[1])
            after_sales["shed"][item] = max(0, int(after_sales["shed"].get(item) or 0) - int(order[2]))
        buys, unmet = top_up_orders(contract, after_sales, max_orders=max(0, 10 - len(core_market) - len(sales)))
        if unmet:
            self.choice["resource_shortages"].append({
                "day": day, "hour": hour, "loc": self.loc,
                "for_day": next_day, "unmet": unmet,
            })
        market = sales + core_market + buys
        if len(market) > 10:
            raise ValueError(f"inverse market has {len(market)} orders at hour {hour}")
        return {"farmer": farmer, "hands": hands, "market": market}

    def _preselect_next_edge(self, obs) -> dict | None:
        if self.preselected_edge is not None:
            return self.preselected_edge
        if self.pending is None:
            return None
        next_id = int(self.pending.get("id") or -1)
        edges = self.net.outgoing.get(next_id) or []
        if not edges:
            return None
        previous_id = self.node_id
        self.node_id = next_id
        try:
            picked = self._select_edge(obs, edges)
        finally:
            self.node_id = previous_id
        self.preselected_edge = picked
        return picked

    def _select_edge(self, obs, edges: list[dict]) -> dict | None:
        if self.path_i < len(self.fixed_path):
            edge = self.fixed_path[self.path_i]
            if int(edge["src"]) != int(self.node_id):
                return None
            self.path_i += 1
            if not self._opened:
                self._opened = True
                self.root_id = int(edge["dst"])
                self.choice["root"] = self.root_id
            return edge
        if self.follow == "random" and edges:
            edge = edges[self.rng.randrange(len(edges))]
            if not self._opened:
                self._opened = True
                self.root_id = int(edge["dst"])
                self.choice["root"] = self.root_id
            return edge
        if not self._opened:
            pinned = [row for row in edges if self.root_id is not None and int(row["dst"]) == int(self.root_id)]
            if pinned:
                edge = pinned[0]
            elif self.greedy:
                edge = edges[0]
            else:
                edge = edges[self.rng.randrange(len(edges))]
            self._opened = True
            self.root_id = int(edge["dst"])
            self.choice["root"] = self.root_id
            return edge
        same = [row for row in edges if self.loc and str(row.get("example") or "") == self.loc]
        if self.greedy and same:
            return same[0]
        if len(edges) == 1:
            return edges[0]
        idx, logp, rec_obs, rec_branches, rec_mask = self._pick(obs, edges)
        edge = edges[idx]
        self.trajectory.append(
            {
                "day": int(obs.get("day") or 0),
                "parent": int(self.node_id),
                "k": len(edges),
                "action": idx,
                "logp": logp,
                "child": int(edge["dst"]),
                "obs": rec_obs,
                "branches": rec_branches,
                "mask": rec_mask,
            }
        )
        self.choice["forks"].append(
            {"day": int(obs.get("day") or 0), "parent": int(self.node_id), "action": idx, "child": int(edge["dst"]), "k": len(edges)}
        )
        return edge

    def _hedge_root(self, edges: list[dict]) -> dict:
        options = {str(int(edge["dst"])): float(self.net.weights.get(str(int(edge["dst"]))) or 1.0) for edge in edges}
        picked = int(sample_program(options, self.rng))
        for edge in edges:
            if int(edge["dst"]) == picked:
                return edge
        return edges[0]

    def _pick(self, obs, edges: list[dict]):
        n = min(len(edges), GRID_MAX_CHOICES)
        chosen = edges[:n]
        obs_vec = encode_obs_from_env(obs)
        branch_vecs = [encode_dest_branch(self.net.dest_of(edge)) for edge in chosen]
        padded, mask = pack_branches(branch_vecs, cap=GRID_MAX_CHOICES)
        if self.policy is None:
            idx = 0 if self.greedy else self.rng.randrange(n)
            return idx, 0.0, obs_vec, padded, mask
        with torch.no_grad():
            o = torch.tensor([obs_vec], dtype=torch.float32, device=self.device)
            b = torch.tensor([padded], dtype=torch.float32, device=self.device)
            m = torch.tensor([mask], dtype=torch.bool, device=self.device)
            logp = self.policy.log_probs(o, b, m)[0, :n]
            if self.greedy:
                idx = int(torch.argmax(logp).item())
            else:
                idx = int(torch.multinomial(torch.exp(logp), 1).item())
            return idx, float(logp[idx].item()), obs_vec, padded, mask


def grid_factory(net: GridNet, policy: NodeScorer | None = None, greedy: bool = True, seed0: int = 0, device: str = "cpu"):
    tapes = TapeBank()
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return GridAgent(net, policy, seed=seed0 + counter["n"] * 17 + 3, greedy=greedy, device=device, tapes=tapes)

    return factory
