"""Load one in-game day's field dispatch from a featured example locator."""
from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import orjson
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from corpus.catalog import day_jobs, read_bytes, zip_member_map
from corpus.extract import _action, _obs

PROD_MARKET = {"HIRE", "BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT", "SELL"}


def effective_field_action(pre_obs: dict, seat: int, action: dict, turns_per_day: int = 24, shed_capacity: int = 100, include_meta: bool = False):
    """Retain only unit actions that actually changed the source state.

    A source PLANT/BUILD blocked by a historical random weed is a no-op. In a
    weed-free replay that same command would otherwise create an extra asset.
    Unit actions are applied in official farmer-then-hands order, including
    atomic same-turn PLANT validation. Market orders remain at their source
    hour; this function only compiles the worker dispatch.
    """
    farm = copy.deepcopy((pre_obs.get("farms") or [{}, {}])[seat])
    private = copy.deepcopy(pre_obs.get("private") or {})
    before_shed = dict(private.get("shed") or {})
    before_seeds = dict(private.get("seeds") or {})
    day = int(pre_obs.get("day") or 0)
    board_size = len(farm.get("tiles") or [])
    raw = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    demand = {}
    for unit in raw:
        if isinstance(unit, list) and len(unit) >= 2 and unit[0] == "PLANT":
            demand[unit[1]] = demand.get(unit[1], 0) + 1
    seeds = private.get("seeds") or {}
    blocked = {crop for crop, n in demand.items() if n > int(seeds.get(crop) or 0)}
    compiled = []
    unit_events = []
    yielded = {}
    for idx, unit in enumerate(raw):
        if not isinstance(unit, list) or not unit:
            compiled.append(["PASS"])
            unit_events.append({"seeds": {}, "shed": {}})
            continue
        if unit[0] == "PASS" or (unit[0] == "PLANT" and len(unit) >= 2 and unit[1] in blocked):
            compiled.append(["PASS"])
            unit_events.append({"seeds": {}, "shed": {}})
            continue
        pos = engine._farmer_position(farm, idx)
        before_pos = tuple(pos) if pos is not None else None
        before_tile = copy.deepcopy(farm["tiles"][pos[1]][pos[0]]) if pos is not None else None
        before_private = copy.deepcopy(private)
        engine._apply_unit_action(farm, private, idx, unit, board_size, day, turns_per_day, shed_capacity)
        seed_before = before_private.get("seeds") or {}
        seed_after = private.get("seeds") or {}
        shed_before = before_private.get("shed") or {}
        shed_after = private.get("shed") or {}
        seed_change = {item: int(seed_after.get(item) or 0) - int(seed_before.get(item) or 0)
                       for item in set(seed_before) | set(seed_after)
                       if int(seed_after.get(item) or 0) != int(seed_before.get(item) or 0)}
        shed_change = {item: int(shed_after.get(item) or 0) - int(shed_before.get(item) or 0)
                       for item in set(shed_before) | set(shed_after)
                       if int(shed_after.get(item) or 0) != int(shed_before.get(item) or 0)}
        unit_events.append({"seeds": seed_change, "shed": shed_change})
        after_pos = engine._farmer_position(farm, idx)
        after_tile = farm["tiles"][pos[1]][pos[0]] if pos is not None else None
        if unit[0] in ("HARVEST", "COLLECT_FERTILIZER"):
            before_bags = before_private.get("inventories") or []
            after_bags = private.get("inventories") or []
            old = before_bags[idx] if idx < len(before_bags) else {}
            new = after_bags[idx] if idx < len(after_bags) else {}
            for item in set(old or {}) | set(new or {}):
                gain = int((new or {}).get(item) or 0) - int((old or {}).get(item) or 0)
                if gain > 0:
                    yielded[item] = int(yielded.get(item) or 0) + gain
        effective = (
            before_pos != (tuple(after_pos) if after_pos is not None else None)
            or before_tile != after_tile
            or before_private != private
        )
        compiled_unit = list(unit) if effective else ["PASS"]
        if effective and len(unit) >= 2 and unit[0] == "PICKUP":
            got = -int(shed_change.get(str(unit[1])) or 0)
            if got > 0:
                compiled_unit = ["PICKUP", unit[1], got]
        elif effective and len(unit) >= 2 and unit[0] == "PLACE":
            deposited = int(shed_change.get(str(unit[1])) or 0)
            if deposited > 0:
                compiled_unit = ["PLACE", unit[1], deposited]
        compiled.append(compiled_unit)
    compiled_action = {
        "farmer": compiled[0] if compiled else ["PASS"],
        "hands": compiled[1:],
        "market": [list(order) for order in (action.get("market") or []) if isinstance(order, list) and order],
    }
    if not include_meta:
        return compiled_action
    after_shed = private.get("shed") or {}
    after_seeds = private.get("seeds") or {}
    meta = {
        "shed_delta_units": {item: int(after_shed.get(item) or 0) - int(before_shed.get(item) or 0) for item in set(before_shed) | set(after_shed)},
        "seed_delta_units": {item: int(after_seeds.get(item) or 0) - int(before_seeds.get(item) or 0) for item in set(before_seeds) | set(after_seeds)},
        "unit_events": unit_events,
        "yielded": yielded,
    }
    return compiled_action, meta


class TapeBank:
    def __init__(self, jobs: list[dict] | None = None, max_cached_episodes: int | None = None):
        self._jobs = {job["date"]: job for job in (day_jobs() if jobs is None else jobs)}
        self.max_cached_episodes = max_cached_episodes
        self._cache_order: OrderedDict[str, None] = OrderedDict()
        self._members: dict[str, dict] = {}
        self._steps: dict[str, list] = {}
        self._metadata: dict[str, dict] = {}
        self._rows: dict[str, list] = {}
        self._dawns: dict[str, dict[int, dict]] = {}

    def _touch(self, loc: str) -> None:
        if self.max_cached_episodes is None:
            return
        self._cache_order[loc] = None
        self._cache_order.move_to_end(loc)
        while len(self._cache_order) > max(1, int(self.max_cached_episodes)):
            old, _ = self._cache_order.popitem(last=False)
            self._steps.pop(old, None)
            self._metadata.pop(old, None)
            self._rows.pop(old, None)
            self._dawns.pop(old, None)

    def _job(self, date: str) -> dict | None:
        return self._jobs.get(date)

    def _load_steps(self, loc: str) -> list:
        if loc in self._steps:
            self._touch(loc)
            return self._steps[loc]
        date, episode, _seat = loc.split(":")
        job = self._job(date)
        if not job:
            self._steps[loc] = []
            self._metadata[loc] = {}
            self._touch(loc)
            return []
        members = None
        if job.get("kind") == "zip" and job.get("archive"):
            archive = str(job["archive"])
            if archive not in self._members:
                self._members[archive] = zip_member_map(Path(archive))
            members = self._members[archive]
        payload = read_bytes(job, episode, members)
        if payload is None:
            self._steps[loc] = []
            self._metadata[loc] = {}
            self._touch(loc)
            return []
        replay = orjson.loads(payload)
        self._metadata[loc] = {
            "info": replay.get("info") or {},
            "configuration": replay.get("configuration") or {},
            "module_version": replay.get("module_version"),
        }
        steps = replay.get("steps") or []
        self._steps[loc] = steps
        self._touch(loc)
        return steps

    def replay_metadata(self, loc: str) -> dict:
        self._load_steps(loc)
        return self._metadata.get(loc, {})

    def _rows_for(self, loc: str) -> list[dict]:
        if loc in self._rows:
            return self._rows[loc]
        steps = self._load_steps(loc)
        seat = int(str(loc).split(":")[2])
        rows: list[dict] = []
        prev = None
        for t in range(max(0, len(steps) - 1)):
            obs = _obs(steps[t], seat, prev)
            if obs is None:
                continue
            post_obs = _obs(steps[t + 1], seat, obs)
            action = _action(steps[t + 1], seat) or {}
            action, meta = effective_field_action(obs, seat, action, include_meta=True)
            market = []
            hires = 0
            for order in action.get("market") or []:
                if not (isinstance(order, list) and order):
                    continue
                op = str(order[0])
                if op not in PROD_MARKET:
                    continue
                market.append(list(order))
                if op == "HIRE":
                    hires += 1
            farmer = action.get("farmer") or ["PASS"]
            if not isinstance(farmer, list) or not farmer:
                farmer = ["PASS"]
            hands = []
            for hand in action.get("hands") or []:
                hands.append(hand if isinstance(hand, list) and hand else ["PASS"])
            rows.append(
                {
                    "day": int(obs.get("day") or 0),
                    "hour": int(obs.get("hour") or 0),
                    "positions": [
                        list((obs.get("farms") or [{}, {}])[seat].get("farmer") or []),
                        *[list(pos) for pos in ((obs.get("farms") or [{}, {}])[seat].get("hands") or [])],
                    ],
                    "land": len((obs.get("farms") or [{}, {}])[seat].get("unlocked_quadrants") or []),
                    "post_hands": len((post_obs.get("farms") or [{}, {}])[seat].get("hands") or []) if post_obs is not None else None,
                    "post_land": len((post_obs.get("farms") or [{}, {}])[seat].get("unlocked_quadrants") or []) if post_obs is not None else None,
                    "shed_delta_units": meta["shed_delta_units"],
                    "seed_delta_units": meta["seed_delta_units"],
                    "unit_events": meta["unit_events"],
                    "yielded": meta["yielded"],
                    "farmer": list(farmer),
                    "hands": hands,
                    "hires": hires,
                    "market": market,
                }
            )
            prev = obs
        self._rows[loc] = rows
        return rows

    def dawn_farm(self, loc: str | None, day: int) -> dict | None:
        if not loc:
            return None
        cached = self._dawns.get(loc)
        if cached is None:
            from mining.grid_net import walk_dawns

            steps = self._load_steps(loc)
            seat = int(str(loc).split(":")[2])
            cached = {int(d): farm for d, farm in walk_dawns(steps, seat)}
            self._dawns[loc] = cached
        return cached.get(int(day))

    def day_tape(self, loc: str | None, day: int) -> list[dict]:
        """Actions to issue during `day`.

        The replay stores each action on the frame that already shows its result.
        `_rows_for` has already shifted it to its pre-action day/hour.
        """
        if not loc:
            return []
        rows = self._rows_for(loc)
        tape: list[dict] = []
        for row in rows:
            if int(row["day"]) != int(day):
                continue
            tape.append(
                {
                    "farmer": list(row["farmer"]),
                    "hands": [list(hand) for hand in row["hands"]],
                    "positions": [list(pos) for pos in row["positions"]],
                    "land": int(row["land"]),
                    "post_hands": row["post_hands"],
                    "post_land": row["post_land"],
                    "shed_delta_units": dict(row["shed_delta_units"]),
                    "seed_delta_units": dict(row["seed_delta_units"]),
                    "unit_events": copy.deepcopy(row["unit_events"]),
                    "yielded": dict(row["yielded"]),
                    "hires": int(row["hires"]),
                    "market": [list(order) for order in row["market"]],
                }
            )
        return tape


def _field_ops(step: dict) -> list:
    ops = [step.get("farmer") or ["PASS"]]
    ops.extend(step.get("hands") or [])
    return ops


def wheat_tape_ok(tape: list[dict], wheat: int) -> bool:
    stock = int(wheat)
    for step in tape or []:
        for order in step.get("market") or []:
            if not order:
                continue
            op = str(order[0])
            qty = int(order[2] or 0) if len(order) >= 3 else 0
            if op == "BUY_PRODUCT" and str(order[1]) == "WHEAT":
                stock += qty
            elif op == "SELL" and str(order[1]) == "WHEAT":
                stock = max(0, stock - qty)
        for op in _field_ops(step):
            if isinstance(op, list) and len(op) >= 3 and op[0] == "PICKUP" and op[1] == "WHEAT":
                qty = int(op[2] or 0)
                if stock < qty:
                    return False
                stock -= qty
    return True


def supply_orders(tape: list[dict], private: dict | None) -> list[list]:
    """Buy whatever today's source field ops will consume beyond stock and the tape's own buys."""
    private = private or {}
    seeds = {k: int(v or 0) for k, v in (private.get("seeds") or {}).items()}
    shed = {k: int(v or 0) for k, v in (private.get("shed") or {}).items()}
    need: dict[tuple, int] = {}

    def add(op: str, item: str, n: int) -> None:
        if n <= 0:
            return
        key = (op, item)
        need[key] = need.get(key, 0) + int(n)

    for step in tape or []:
        for action in _field_ops(step):
            if not (isinstance(action, list) and action):
                continue
            if action[0] == "PLANT" and len(action) >= 2:
                crop = str(action[1])
                if seeds.get(crop, 0) <= 0:
                    add("BUY_SEED", crop, 1)
                    seeds[crop] = 1
                seeds[crop] = seeds.get(crop, 0) - 1
            elif action[0] == "PICKUP" and len(action) >= 2:
                item = str(action[1])
                qty = int(action[2] or 1) if len(action) >= 3 else 1
                have = shed.get(item, 0)
                if have < qty:
                    gap = qty - have
                    if item in ("WHEAT", "FERTILIZER"):
                        add("BUY_PRODUCT", item, gap)
                    else:
                        add("BUY_ANIMAL", item, gap)
                    shed[item] = qty
                shed[item] = shed.get(item, 0) - qty
    orders = []
    for (op, item), qty in need.items():
        orders.append([op, item, int(qty)])
    return orders
