"""Dawn farm-grid identity. Same tiles at the same day merge into one net node."""
from __future__ import annotations

import hashlib

from config import ANIMALS, CROPS


def tile_cell(tile, day: int) -> str:
    if tile is None:
        return "."
    if tile == "LOCKED":
        return "L"
    if tile == "WEED":
        return "w"
    if not isinstance(tile, dict):
        return "?"
    kind = tile.get("kind")
    if kind == "WEED":
        return "w"
    if kind == "PLANT":
        crop = str(tile.get("crop") or "?")
        age = max(0, int(day) - int(tile.get("planted_day") or 0))
        ready = "r" if int(tile.get("yield_units") or 0) > 0 else ""
        return f"p{crop[0]}{age}{ready}"
    if kind in ("PASTURE", "COOP"):
        animal = tile.get("animal")
        mark = "u" if kind == "PASTURE" else "o"
        if not animal:
            return mark
        unfed = int(tile.get("consecutive_unfed") or 0)
        return f"{mark}{animal[0]}{unfed}"
    return "?"


def grid_text(farm: dict, day: int) -> str:
    rows = []
    for row in farm.get("tiles") or []:
        rows.append("".join(tile_cell(tile, day) for tile in (row or [])))
    return "/".join(rows)


def grid_hash(farm: dict, day: int) -> str:
    return hashlib.sha1(grid_text(farm, day).encode("utf-8")).hexdigest()[:16]


_DAWN_RESET = {"watered_today", "fed_today", "cared_today"}


def exact_cell(tile) -> str:
    """Full dawn tile. Fields cleared every midnight are not part of identity."""
    if tile is None:
        return "."
    if tile == "LOCKED":
        return "L"
    if not isinstance(tile, dict):
        return str(tile)
    parts = []
    for key in sorted(tile):
        if key in _DAWN_RESET:
            continue
        parts.append(f"{key}={tile[key]}")
    return ",".join(parts)


def exact_text(farm: dict) -> str:
    rows = []
    for row in farm.get("tiles") or []:
        rows.append("|".join(exact_cell(tile) for tile in (row or [])))
    return "/".join(rows)


def exact_hash(farm: dict) -> str:
    return hashlib.sha1(exact_text(farm).encode("utf-8")).hexdigest()[:16]


def production_cell_match(live, dest) -> bool:
    """Match a dawn production tile while accepting strictly surplus yield.

    Weed repair can make an originally ineffective WATER/CARE action succeed.
    In that case the live tile is structurally identical to the witness tile but
    may hold more harvestable units.  That is a safe state for continuing the
    witness schedule; a deficit is not.  All non-yield fields remain exact.
    """
    if not isinstance(live, dict) or not isinstance(dest, dict):
        return exact_cell(live) == exact_cell(dest)
    live_core = {
        key: value for key, value in live.items()
        if key not in _DAWN_RESET and key != "yield_units"
    }
    dest_core = {
        key: value for key, value in dest.items()
        if key not in _DAWN_RESET and key != "yield_units"
    }
    if live_core != dest_core:
        return False
    return int(live.get("yield_units") or 0) >= int(dest.get("yield_units") or 0)


def is_weed(tile) -> bool:
    if tile == "WEED":
        return True
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def is_plant(tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_idle(tile) -> bool:
    """Empty or random-weed overlay. Not a production asset."""
    return tile is None or is_weed(tile)


def weed_overlay(farm: dict) -> list[list[int]]:
    out = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row or []):
            if is_weed(tile):
                out.append([x, y])
    return out


def prod_cell(tile) -> str:
    """Candidate-index cell: idle land (empty or weed) is the same slot.

    Production assets keep full exact identity. Crop that became weed is still
    a weed tile here; callers that have yesterday's tile must use prod_match.
    """
    if is_idle(tile):
        return "."
    return exact_cell(tile)


def prod_text(farm: dict) -> str:
    land = ",".join(str(q) for q in (farm.get("unlocked_quadrants") or []))
    rows = []
    for row in farm.get("tiles") or []:
        rows.append("|".join(prod_cell(tile) for tile in (row or [])))
    return land + "//" + "/".join(rows)


def prod_hash(farm: dict) -> str:
    """Recall key: random weeds on idle tiles do not change identity."""
    return hashlib.sha1(prod_text(farm).encode("utf-8")).hexdigest()[:16]


def prod_match(live: dict, dest: dict, dest_prev: dict | None = None) -> bool:
    """Compare dawn production assets, ignoring the idle weed overlay.

    Yesterday's plant does not prove that today's weed is a decayed crop: the
    plant may have been harvested today and a random weed spawned afterward.
    Both outcomes are non-producing idle land. A lost *living* plant remains a
    mismatch, and the first day it disappears is caught when the target still
    contains that plant.
    """
    if sorted(live.get("unlocked_quadrants") or []) != sorted(dest.get("unlocked_quadrants") or []):
        return False
    lt = live.get("tiles") or []
    dt = dest.get("tiles") or []
    if len(lt) != len(dt):
        return False
    for y, (lrow, drow) in enumerate(zip(lt, dt)):
        lrow = lrow or []
        drow = drow or []
        if len(lrow) != len(drow):
            return False
        for x, (lv, dv) in enumerate(zip(lrow, drow)):
            if is_idle(dv) and is_idle(lv):
                continue
            if not production_cell_match(lv, dv):
                return False
    return True


def occup_from_farm(farm: dict) -> dict:
    crops = {name: 0 for name in CROPS}
    animals = {name: 0 for name in ANIMALS}
    for row in farm.get("tiles") or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in crops:
                    crops[crop] += 1
            if tile.get("animal") in animals:
                animals[tile["animal"]] += 1
    return {
        "crops": crops,
        "animals": animals,
        "land": len(farm.get("unlocked_quadrants") or []),
    }


def materials_needed(stats: dict, private: dict, dest: dict) -> list[list]:
    """Shopping list for a chosen next farm state. Not part of the forest topology."""
    seeds = private.get("seeds") or {}
    shed = private.get("shed") or {}
    dest_crops = dest.get("crops") or {}
    dest_animals = dest.get("animals") or {}
    dest_land = int(dest.get("land") or 1)
    live_c = stats.get("crops") or {}
    live_a = stats.get("animals") or {}
    live_land = int(stats.get("land") or 1)
    orders: list[list] = []
    for name in CROPS:
        gap = int(dest_crops.get(name) or 0) - int(live_c.get(name) or 0) - int(seeds.get(name) or 0)
        if gap > 0:
            orders.append(["BUY_SEED", name, gap])
    for name in ANIMALS:
        gap = int(dest_animals.get(name) or 0) - int(live_a.get(name) or 0) - int(shed.get(name) or 0)
        if gap > 0:
            orders.append(["BUY_ANIMAL", name, gap])
    for _ in range(max(0, dest_land - live_land)):
        orders.append(["BUY_LAND"])
    herd = sum(int(dest_animals.get(name) or 0) for name in ANIMALS)
    wheat = int(shed.get("WHEAT") or 0)
    if herd and wheat < herd:
        orders.append(["BUY_PRODUCT", "WHEAT", herd - wheat])
    return orders


PROD_MARKET = ("BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND", "BUY_PRODUCT")


def pack_schedule(orders: dict[tuple, int]) -> str:
    parts = []
    for (op, item), qty in sorted(orders.items()):
        if int(qty or 0) <= 0:
            continue
        if op == "HIRE":
            parts.append(f"HIRE:{int(qty)}")
        elif op == "BUY_LAND":
            parts.append(f"LAND:{int(qty)}")
        else:
            parts.append(f"{op}:{item}:{int(qty)}")
    return "|".join(parts) or "NOP"


def unpack_schedule(text: str) -> list[list]:
    out = []
    if not text or text == "NOP":
        return out
    for part in text.split("|"):
        bits = part.split(":")
        if bits[0] == "HIRE":
            n = int(bits[1] or 0)
            out.extend([["HIRE"] for _ in range(n)])
        elif bits[0] == "LAND":
            n = int(bits[1] or 0)
            out.extend([["BUY_LAND"] for _ in range(n)])
        elif len(bits) >= 3:
            out.append([bits[0], bits[1], int(bits[2] or 0)])
    return out


def add_order(bucket: dict[tuple, int], order: list) -> None:
    if not order:
        return
    op = str(order[0])
    if op not in PROD_MARKET:
        return
    if op == "HIRE":
        key = ("HIRE", "")
        bucket[key] = int(bucket.get(key) or 0) + 1
        return
    if op == "BUY_LAND":
        key = ("BUY_LAND", "")
        bucket[key] = int(bucket.get(key) or 0) + 1
        return
    if len(order) >= 3:
        item = str(order[1])
        qty = int(order[2] or 0)
        key = (op, item)
        bucket[key] = int(bucket.get(key) or 0) + qty
