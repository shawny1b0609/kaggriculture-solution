"""State-guarded program executor. Recipes from featured families, not a 1911 tape.

Hands are dismissed at end of every day. hire_daily is a *daily* rehire count,
not a season-long headcount. Featured 1.32.7 winners finish near 9 workers on
the last day, 3-4 land, and a cow/sheep ranch.
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as ANIMAL_RULES,
    CROPS as CROP_RULES,
    LAND_PRICES,
    _hire_cost,
    market_price,
)

from config import CROPS
from runtime.seller import sell
from runtime.stock import clip_market
from runtime.view import (
    counts,
    nearest_shed,
    own_farm,
    shed_tiles,
    step_towards,
    tile_iter,
    worker_positions,
)

LAST_PLANT = {
    "WHEAT": 27,
    "CARROT": 27,
    "TOMATO": 20,
    "STRAWBERRY": 18,
    "MELON": 17,
}


def default_programs() -> dict[str, dict]:
    """Bootstrap cards; mining overwrites from full 1.32.7 scan.

    Opening dump and late mix taken from 2026-08-16 featured winner medians
    (reconstruction checkpoint snaps, evidence only).
    """
    return {
        "wheat_cow": {
            "program_id": "wheat_cow",
            "crops": {"WHEAT": 16, "MELON": 12, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 20},
            "animals": {"COW": 10, "SHEEP": 4, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 3,
            "reserve_wheat": 24,
            "open_animals": {"COW": 2, "SHEEP": 2, "GOOSE": 0},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "PET_CAFE": "carrot_cycle"},
        },
        "wheat_sheep": {
            "program_id": "wheat_sheep",
            "crops": {"WHEAT": 16, "MELON": 8, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 12},
            "animals": {"COW": 4, "SHEEP": 10, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 3,
            "reserve_wheat": 24,
            "open_animals": {"COW": 2, "SHEEP": 2, "GOOSE": 0},
            "shop_switch": {"PIZZA_SHOP": "wheat_cow", "ICE_CREAM_SHOP": "berry_milk"},
        },
        "goose_wheat": {
            "program_id": "goose_wheat",
            "crops": {"WHEAT": 14, "MELON": 8, "CARROT": 4, "TOMATO": 0, "STRAWBERRY": 8},
            "animals": {"COW": 2, "SHEEP": 0, "GOOSE": 10},
            "hire_daily": 7,
            "land_target": 3,
            "reserve_wheat": 20,
            "open_animals": {"COW": 0, "SHEEP": 0, "GOOSE": 2},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "PIZZA_SHOP": "wheat_cow"},
        },
        "melon_field": {
            "program_id": "melon_field",
            "crops": {"WHEAT": 8, "MELON": 16, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 8},
            "animals": {"COW": 6, "SHEEP": 2, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 3,
            "reserve_wheat": 16,
            "open_animals": {"COW": 2, "SHEEP": 0, "GOOSE": 0},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "PIZZA_SHOP": "tomato_milk"},
        },
        "tomato_milk": {
            "program_id": "tomato_milk",
            "crops": {"WHEAT": 12, "MELON": 6, "CARROT": 0, "TOMATO": 12, "STRAWBERRY": 8},
            "animals": {"COW": 10, "SHEEP": 2, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 3,
            "reserve_wheat": 22,
            "open_animals": {"COW": 2, "SHEEP": 0, "GOOSE": 0},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "SMOOTHIE_SHOP": "berry_milk"},
        },
        "berry_milk": {
            "program_id": "berry_milk",
            "crops": {"WHEAT": 10, "MELON": 4, "CARROT": 0, "TOMATO": 4, "STRAWBERRY": 24},
            "animals": {"COW": 8, "SHEEP": 2, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 4,
            "reserve_wheat": 20,
            "open_animals": {"COW": 2, "SHEEP": 0, "GOOSE": 0},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "PET_CAFE": "carrot_cycle"},
        },
        "carrot_cycle": {
            "program_id": "carrot_cycle",
            "crops": {"WHEAT": 8, "MELON": 6, "CARROT": 16, "TOMATO": 0, "STRAWBERRY": 4},
            "animals": {"COW": 4, "SHEEP": 2, "GOOSE": 2},
            "hire_daily": 7,
            "land_target": 3,
            "reserve_wheat": 14,
            "open_animals": {"COW": 2, "SHEEP": 0, "GOOSE": 0},
            "shop_switch": {"PIZZA_SHOP": "tomato_milk", "YARN_STORE": "wheat_sheep"},
        },
        "mixed_balance": {
            "program_id": "mixed_balance",
            "crops": {"WHEAT": 14, "MELON": 8, "CARROT": 2, "TOMATO": 4, "STRAWBERRY": 12},
            "animals": {"COW": 6, "SHEEP": 6, "GOOSE": 0},
            "hire_daily": 8,
            "land_target": 3,
            "reserve_wheat": 22,
            "open_animals": {"COW": 2, "SHEEP": 2, "GOOSE": 0},
            "shop_switch": {"YARN_STORE": "wheat_sheep", "PET_CAFE": "carrot_cycle"},
        },
    }


def _hire_total(n: int) -> int:
    return sum(_hire_cost(i) for i in range(max(0, n)))


class ProgramVM:
    def __init__(self, card: dict):
        self.card = card
        self.program_id = card["program_id"]
        self.calls = 0
        self.choice = {"program": self.program_id}
        self._seed_budget = {}
        self._bought_animals_today = 0
        self._market_day = -1
        self._guided = "day_schedule" in card
        self._shop_queue = [list(order) for order in (card.get("day_schedule") or [])]

    def __call__(self, obs, cfg=None):
        self.calls += 1
        farm = own_farm(obs)
        private = obs.get("private") or {}
        self._seed_budget = {k: int(v or 0) for k, v in (private.get("seeds") or {}).items()}
        stats = counts(farm)
        day = int(obs.get("day") or 0)
        hour = int(obs.get("hour") or 0)
        positions = worker_positions(farm)
        n_workers = len(positions)
        inventories = list(private.get("inventories") or [{}])
        while len(inventories) < n_workers:
            inventories.append({})
        tasks = self._tasks(obs, farm, private, stats, day, n_workers)
        field = []
        claimed = set()
        for idx, pos in enumerate(positions):
            bag = inventories[idx] if idx < len(inventories) else {}
            action = self._act(pos, bag, tasks, claimed, farm, private, stats, day)
            field.append(action)
        market = self._market(obs, farm, private, stats, day, hour)
        return {"farmer": field[0] if field else ["PASS"], "hands": field[1:], "market": market}

    def _need_crop(self, stats, day) -> str | None:
        wheat = int(stats["crops"].get("WHEAT") or 0)
        if wheat < 4 and day <= LAST_PLANT["WHEAT"] and int(self.card["crops"].get("WHEAT") or 0) > 0:
            return "WHEAT"
        best = None
        best_gap = 0
        for crop, target in self.card["crops"].items():
            if int(target or 0) <= 0:
                continue
            if day > LAST_PLANT.get(crop, 27):
                continue
            gap = int(target) - int(stats["crops"].get(crop) or 0)
            if gap > best_gap:
                best_gap = gap
                best = crop
        return best

    def _animal_targets(self, day) -> dict[str, int]:
        if day < 4:
            return {k: int(v or 0) for k, v in (self.card.get("open_animals") or {}).items()}
        targets = {k: int(v or 0) for k, v in (self.card.get("open_animals") or {}).items()}
        for k, v in (self.card.get("animals") or {}).items():
            targets[k] = max(int(targets.get(k) or 0), int(v or 0))
        return targets

    def _need_animal(self, stats, day, shed=None) -> str | None:
        """Missing animals on the board. Shed stock still needs a structure and PLACE."""
        for animal, target in self._animal_targets(day).items():
            if int(target or 0) <= 0:
                continue
            if int(stats["animals"].get(animal, 0) or 0) < int(target):
                return animal
        return None

    def _want_struct(self, farm, stats, day, private) -> tuple[str | None, str | None, int]:
        animal = self._need_animal(stats, day, private.get("shed") or {})
        if not animal:
            return None, None, 0
        structure = ANIMAL_RULES[animal]["structure"]
        target_n = int(self._animal_targets(day).get(animal) or 0)
        existing = self._structure_count(farm, structure)
        empty_struct = stats["empty_pasture"] if structure == "PASTURE" else stats["empty_coop"]
        missing_animals = max(0, target_n - int(stats["animals"].get(animal, 0) or 0))
        want = max(0, missing_animals - empty_struct, target_n - existing)
        return animal, structure, min(want, 3)

    def _structure_count(self, farm, structure: str) -> int:
        n = 0
        for _, _, tile in tile_iter(farm):
            if isinstance(tile, dict) and tile.get("kind") == structure:
                n += 1
        return n

    def _tasks(self, obs, farm, private, stats, day, n_workers) -> list[dict]:
        tasks = []
        seeds = private.get("seeds") or {}
        animal, structure, want_struct = self._want_struct(farm, stats, day, private)
        shed = private.get("shed") or {}
        wheat_ok = int(stats["crops"].get("WHEAT") or 0) >= 4 or int(shed.get("WHEAT") or 0) >= 2 or day >= 1
        animal_waiting = bool(animal) and (
            int(shed.get(animal) or 0) > 0 or sum(int(shed.get(a) or 0) for a in ANIMAL_RULES) > 0 or day >= 0
        )
        builds = 0
        plants = 0
        plant_cap = max(8, n_workers * 3)
        seed_left = {k: int(v or 0) for k, v in seeds.items()}
        for x, y, tile in tile_iter(farm):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                age = day - int(tile.get("planted_day") or 0)
                first = CROP_RULES.get(crop, {}).get("first_yield_day", 0)
                if int(tile.get("yield_units") or 0) > 0 and age >= first:
                    tasks.append({"kind": "HARVEST", "at": (x, y), "pri": 16 if day >= 28 else 13})
                elif not tile.get("watered_today"):
                    tasks.append({"kind": "WATER", "at": (x, y), "pri": 12})
            if isinstance(tile, dict) and tile.get("animal"):
                if int(tile.get("yield_units") or 0) > 0:
                    tasks.append({"kind": "HARVEST", "at": (x, y), "pri": 17})
                if not tile.get("fed_today"):
                    tasks.append({"kind": "FEED", "at": (x, y), "pri": 18})
                if not tile.get("cared_today"):
                    tasks.append({"kind": "CARE", "at": (x, y), "pri": 4})
                if tile.get("fertilizer_available"):
                    tasks.append({"kind": "COLLECT_FERTILIZER", "at": (x, y), "pri": 3})
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append({"kind": "DIG", "at": (x, y), "pri": 2})
            if animal and isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal"):
                tasks.append({"kind": "PLACE_ANIMAL", "at": (x, y), "item": animal, "pri": 15})
            if tile is None:
                crop = self._need_crop(stats, day)
                prefer_build = animal_waiting and wheat_ok and builds < want_struct
                if prefer_build and plants >= 4:
                    tasks.append({"kind": "BUILD", "at": (x, y), "structure": structure, "pri": 14})
                    builds += 1
                elif crop and int(seed_left.get(crop) or 0) > 0 and plants < plant_cap and not (prefer_build and plants >= 4):
                    tasks.append({"kind": "PLANT", "at": (x, y), "crop": crop, "pri": 9 if plants < 6 else 7})
                    seed_left[crop] = int(seed_left.get(crop) or 0) - 1
                    plants += 1
                elif prefer_build:
                    tasks.append({"kind": "BUILD", "at": (x, y), "structure": structure, "pri": 14})
                    builds += 1
                elif crop and int(seed_left.get(crop) or 0) > 0 and plants < plant_cap:
                    tasks.append({"kind": "PLANT", "at": (x, y), "crop": crop, "pri": 7})
                    seed_left[crop] = int(seed_left.get(crop) or 0) - 1
                    plants += 1
        tasks.sort(key=lambda t: -t["pri"])
        return tasks

    def _act(self, pos, bag, tasks, claimed, farm, private, stats, day):
        shed = private.get("shed") or {}
        hungry = int(stats["hungry"] or 0) > 0
        wheat_keep = 0
        if hungry or sum(int(stats["animals"].get(a, 0) or 0) for a in ANIMAL_RULES) or sum(int(shed.get(a) or 0) for a in ANIMAL_RULES):
            wheat_keep = max(int(stats["hungry"] or 0), 2)
        extra_wheat = max(0, int(bag.get("WHEAT") or 0) - wheat_keep)
        other = 0
        cargo = []
        for item, qty in (bag or {}).items():
            q = int(qty or 0)
            if q <= 0 or item in ANIMAL_RULES:
                continue
            if item == "WHEAT":
                if extra_wheat <= 0:
                    continue
                cargo.append("WHEAT")
                other += extra_wheat
                continue
            cargo.append(item)
            other += q
        if cargo and other >= 1:
            dest = nearest_shed(pos)
            if pos == dest:
                item = cargo[0]
                qty = extra_wheat if item == "WHEAT" else int(bag.get(item) or 1)
                return ["PLACE", item, max(1, qty)]
            return step_towards(pos, dest)
        if hungry and int(bag.get("WHEAT") or 0) == 0 and int(shed.get("WHEAT") or 0) > 0:
            dest = nearest_shed(pos)
            if pos == dest:
                return ["PICKUP", "WHEAT", min(4, max(1, int(stats["hungry"] or 1)), int(shed["WHEAT"]))]
            return step_towards(pos, dest)
        animal = self._need_animal(stats, day, private.get("shed") or {})
        if (
            animal
            and (not hungry or int(bag.get("WHEAT") or 0) > 0)
            and pos in shed_tiles()
            and int(shed.get(animal) or 0) > 0
            and int(bag.get(animal) or 0) == 0
        ):
            structure = ANIMAL_RULES[animal]["structure"]
            empty_struct = stats["empty_pasture"] if structure == "PASTURE" else stats["empty_coop"]
            if empty_struct > 0:
                return ["PICKUP", animal, 1]
        carried_animal = next((name for name in ANIMAL_RULES if int(bag.get(name) or 0) > 0), None)
        if carried_animal:
            structure = ANIMAL_RULES[carried_animal]["structure"]
            empty_struct = stats["empty_pasture"] if structure == "PASTURE" else stats["empty_coop"]
            if empty_struct <= 0:
                dest = nearest_shed(pos)
                if pos == dest:
                    return ["PLACE", carried_animal, 1]
                return step_towards(pos, dest)
        tiles = farm.get("tiles") or []
        x, y = pos
        tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None
        here = self._local_action(tile, bag, stats, day, private)
        if here and hungry and here[0] == "PLANT":
            if len(here) < 2 or here[1] != "WHEAT":
                here = None
        if here and hungry and here[0] in ("DIG", "CARE", "COLLECT_FERTILIZER", "BUILD_PASTURE", "BUILD_COOP"):
            here = None
        if here and hungry and here[0] == "WATER":
            crop = tile.get("crop") if isinstance(tile, dict) else None
            if crop != "WHEAT":
                here = None
        if here:
            return here
        for task in tasks:
            dest = task["at"]
            if dest in claimed:
                continue
            if task["kind"] == "FEED" and int(bag.get("WHEAT") or 0) <= 0:
                if int(shed.get("WHEAT") or 0) > 0:
                    dest = nearest_shed(pos)
                else:
                    continue
            if task["kind"] == "PLACE_ANIMAL" and int(bag.get(task.get("item") or "") or 0) <= 0:
                if hungry:
                    continue
                if int(shed.get(task.get("item") or "") or 0) > 0:
                    dest = nearest_shed(pos)
                else:
                    continue
            if task["kind"] == "PLANT":
                crop = task.get("crop")
                if int((private.get("seeds") or {}).get(crop) or 0) <= 0:
                    continue
            claimed.add(task["at"])
            if dest == pos:
                return self._task_action(task, bag)
            return step_towards(pos, dest)
        if cargo:
            dest = nearest_shed(pos)
            if pos == dest:
                item = cargo[0]
                return ["PLACE", item, int(bag.get(item) or 1)]
            return step_towards(pos, dest)
        return ["PASS"]

    def _local_action(self, tile, bag, stats, day, private):
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            age = day - int(tile.get("planted_day") or 0)
            first = CROP_RULES.get(crop, {}).get("first_yield_day", 0)
            if int(tile.get("yield_units") or 0) > 0 and age >= first:
                return ["HARVEST"]
            if not tile.get("watered_today"):
                return ["WATER"]
        if isinstance(tile, dict) and tile.get("animal"):
            if int(tile.get("yield_units") or 0) > 0:
                return ["HARVEST"]
            if not tile.get("fed_today") and int(bag.get("WHEAT") or 0) > 0:
                return ["FEED"]
            if not tile.get("cared_today"):
                return ["CARE"]
            if tile.get("fertilizer_available"):
                return ["COLLECT_FERTILIZER"]
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ["DIG"]
        animal = self._need_animal(stats, day, private.get("shed") or {})
        if animal and isinstance(tile, dict) and tile.get("kind") == ANIMAL_RULES[animal]["structure"] and not tile.get("animal"):
            if int(bag.get(animal) or 0) > 0:
                return ["PLACE", animal]
        if tile is None:
            shed = private.get("shed") or {}
            wheat_ok = int(stats["crops"].get("WHEAT") or 0) >= 4 or int(shed.get("WHEAT") or 0) >= 2 or day >= 1
            if animal and wheat_ok:
                structure = ANIMAL_RULES[animal]["structure"]
                empty_struct = stats["empty_pasture"] if structure == "PASTURE" else stats["empty_coop"]
                target_n = int(self._animal_targets(day).get(animal) or 0)
                if empty_struct <= 0 and target_n > 0:
                    return ["BUILD_PASTURE"] if structure == "PASTURE" else ["BUILD_COOP"]
            crop = self._need_crop(stats, day)
            if crop and int(self._seed_budget.get(crop) or 0) > 0:
                self._seed_budget[crop] = int(self._seed_budget.get(crop) or 0) - 1
                return ["PLANT", crop]
        return None

    def _task_action(self, task, bag):
        kind = task["kind"]
        if kind == "PLANT":
            crop = task["crop"]
            if int(self._seed_budget.get(crop) or 0) <= 0:
                return ["PASS"]
            self._seed_budget[crop] = int(self._seed_budget.get(crop) or 0) - 1
            return ["PLANT", crop]
        if kind == "BUILD":
            return ["BUILD_PASTURE"] if task["structure"] == "PASTURE" else ["BUILD_COOP"]
        if kind == "PLACE_ANIMAL":
            return ["PLACE", task["item"]] if int(bag.get(task.get("item") or "") or 0) > 0 else ["PASS"]
        if kind == "FEED":
            return ["FEED"] if int(bag.get("WHEAT") or 0) > 0 else ["PASS"]
        return [kind]

    def _append_hires(self, orders, cash, already, hire_n, room: int):
        added = 0
        for i in range(max(0, hire_n - already)):
            if added >= room:
                break
            cost = _hire_cost(already + i)
            if cash < cost:
                break
            if already + i >= 4 and cash < cost + 20:
                break
            orders.append(["HIRE"])
            cash -= cost
            added += 1
        return cash

    def _market(self, obs, farm, private, stats, day, hour) -> list:
        if day != self._market_day:
            self._market_day = day
            self._bought_animals_today = 0
        cash = float(farm.get("money") or 0)
        shed = dict(private.get("shed") or {})
        seeds = private.get("seeds") or {}
        hire_n = int(self.card.get("hire_daily") or 8)
        already = int(farm.get("hires_today") or 0)
        extra = stats["land"] - 1
        land_target = int(self.card.get("land_target") or 1)
        animal_targets = self._animal_targets(day)
        missing_animal_cost = 0
        for animal, target in animal_targets.items():
            have = int(stats["animals"].get(animal, 0) or 0) + int(shed.get(animal) or 0)
            missing = max(0, int(target or 0) - have)
            if day > 20:
                missing = 0
            missing_animal_cost += missing * int(ANIMAL_RULES[animal]["cost"])
        dump = []
        if stats["land"] < land_target and extra < len(LAND_PRICES) and day >= 2:
            price = LAND_PRICES[extra]
            if cash > price + 500 + min(missing_animal_cost, 800):
                dump.append(["BUY_LAND"])
                cash -= price
        hire_reserve = 54 if already < hire_n or hour >= 20 else 12
        feed_reserve = 40 if missing_animal_cost or stats["hungry"] else 0
        cash_floor = max(hire_reserve, 12)
        seed_cash = max(0, cash - missing_animal_cost - cash_floor - feed_reserve)
        for crop, target in self.card["crops"].items():
            if int(target or 0) <= 0:
                continue
            if day > LAST_PLANT.get(crop, 27):
                continue
            if day == 0 and crop in ("STRAWBERRY", "TOMATO"):
                continue
            plant_have = int(seeds.get(crop) or 0) + int(stats["crops"].get(crop, 0) or 0)
            want = int(target)
            if day == 0:
                want = min(want, {"WHEAT": 8, "MELON": 12, "CARROT": 4}.get(crop, want))
            missing = max(0, want - plant_have)
            if missing <= 0:
                continue
            unit = CROP_RULES[crop]["seed"]
            cap = 8 if day == 0 else 6
            buy = min(missing, cap, int(max(0, seed_cash) // unit))
            if buy > 0:
                dump.append(["BUY_SEED", crop, buy])
                cash -= buy * unit
                seed_cash -= buy * unit
        bought_n = 0
        for animal, target in animal_targets.items():
            have = int(stats["animals"].get(animal, 0) or 0) + int(shed.get(animal) or 0)
            missing = max(0, int(target or 0) - have)
            if missing <= 0 or day > 20:
                continue
            unit = ANIMAL_RULES[animal]["cost"]
            if day > 0 and (stats["hungry"] > 0 or self._bought_animals_today >= 1 or cash < unit + 100):
                continue
            n = 2 if day == 0 else 1
            n = min(n, missing, int(max(0, cash - cash_floor - feed_reserve) // unit))
            if day > 0:
                n = min(n, 1)
            if n > 0:
                dump.append(["BUY_ANIMAL", animal, n])
                cash -= n * unit
                bought_n += n
                self._bought_animals_today += n
            if bought_n >= 4:
                break
        hungry = stats["hungry"]
        pending_animals = sum(int(shed.get(a) or 0) for a in ANIMAL_RULES) + bought_n + sum(stats["animals"].values())
        wheat = int(shed.get("WHEAT") or 0)
        feed_need = max(hungry + pending_animals, 4 if pending_animals else 0)
        if feed_need and wheat < feed_need:
            inv = int((obs.get("market") or {}).get("inventory", {}).get("WHEAT") or 0)
            price = market_price("WHEAT", inv, (obs.get("market") or {}).get("params"))
            need = min(max(1, feed_need - wheat), 8)
            if cash > price * need + cash_floor:
                dump.append(["BUY_PRODUCT", "WHEAT", need])
                cash -= price * need
        available = {k: int(shed.get(k) or 0) for k in shed}
        wheat_keep = max(stats["hungry"] + pending_animals, 4)
        if cash >= 800:
            wheat_keep = max(wheat_keep, min(int(self.card.get("reserve_wheat") or 8), 12))
        extra_wheat = max(0, int(shed.get("WHEAT") or 0) - wheat_keep)
        if extra_wheat:
            available["WHEAT"] = extra_wheat
            reserved_w = {"WHEAT": wheat_keep}
        else:
            reserved_w = {"WHEAT": int(shed.get("WHEAT") or 0)}
        sell_orders = sell(obs, available, reserved_w)
        if extra_wheat and not any(o[0] == "SELL" and o[1] == "WHEAT" for o in sell_orders):
            sell_orders.append(["SELL", "WHEAT", extra_wheat])
        if self._guided:
            dump = []
            dump_cap = 6 if hour == 0 else 4
            while self._shop_queue and len(dump) < dump_cap:
                dump.append(self._shop_queue.pop(0))
            hire_orders = []
            room = max(0, 10 - len(dump) - min(len(sell_orders), 2 if hour == 0 else 6))
            if hire_n > already and room:
                cash = self._append_hires(hire_orders, cash, already, hire_n, room)
            mixed = dump + hire_orders + sell_orders[: 2 if hour == 0 else 6]
            return clip_market(mixed[:10], farm, private, obs)
        dump_cap = 6 if hour == 0 else 5
        dump = dump[:dump_cap]
        sell_cap = 6 if hour != 0 else 2
        sell_orders = sell_orders[:sell_cap]
        hire_orders = []
        room = max(0, 10 - len(dump) - len(sell_orders))
        if hire_n > already and room:
            cash = self._append_hires(hire_orders, cash, already, hire_n, room)
        if hour == 0:
            mixed = dump + hire_orders + sell_orders
        else:
            mixed = hire_orders[:3] + dump + sell_orders + hire_orders[3:]
        return clip_market(mixed, farm, private, obs)
