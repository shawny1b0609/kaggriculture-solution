"""Legal observation helpers. No opponent private, no future shops, no seed."""
from __future__ import annotations

from config import ANIMALS, CROPS, PRODUCTS


def step_of(obs) -> int:
    if obs.get("step") is not None:
        return int(obs["step"])
    return int(obs.get("day") or 0) * 24 + int(obs.get("hour") or 0)


def own_farm(obs) -> dict:
    seat = int(obs["player"])
    return obs["farms"][seat]


def rival_farm(obs) -> dict:
    seat = int(obs["player"])
    farms = obs["farms"]
    return farms[1 - seat] if len(farms) > 1 else {}


def tile_iter(farm):
    tiles = farm.get("tiles") or []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row or []):
            yield x, y, tile


def counts(farm) -> dict:
    crops = {name: 0 for name in CROPS}
    animals = {name: 0 for name in ANIMALS}
    empty = weeds = locked = ready_crop = ready_animal = hungry = unwatered = empty_pasture = empty_coop = 0
    for x, y, tile in tile_iter(farm):
        if tile is None:
            empty += 1
            continue
        if tile == "LOCKED":
            locked += 1
            continue
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            weeds += 1
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            if crop in crops:
                crops[crop] += 1
            if int(tile.get("yield_units") or 0) > 0:
                ready_crop += 1
            if not tile.get("watered_today"):
                unwatered += 1
        elif isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal"):
            empty_pasture += 1
        elif isinstance(tile, dict) and tile.get("kind") == "COOP" and not tile.get("animal"):
            empty_coop += 1
        if isinstance(tile, dict) and tile.get("animal"):
            name = tile["animal"]
            if name in animals:
                animals[name] += 1
            if not tile.get("fed_today"):
                hungry += 1
            if int(tile.get("yield_units") or 0) > 0:
                ready_animal += 1
    return {
        "crops": crops,
        "animals": animals,
        "empty": empty,
        "weeds": weeds,
        "locked": locked,
        "ready_crop": ready_crop,
        "ready_animal": ready_animal,
        "hungry": hungry,
        "unwatered": unwatered,
        "empty_pasture": empty_pasture,
        "empty_coop": empty_coop,
        "workers": 1 + len(farm.get("hands") or []),
        "land": len(farm.get("unlocked_quadrants") or []),
        "cash": float(farm.get("money") or 0),
        "hires_today": int(farm.get("hires_today") or 0),
    }


def bag_total(private) -> dict[str, int]:
    out = {item: 0 for item in PRODUCTS}
    for item, qty in (private.get("shed") or {}).items():
        out[item] = out.get(item, 0) + int(qty or 0)
    for bag in private.get("inventories") or []:
        for item, qty in (bag or {}).items():
            out[item] = out.get(item, 0) + int(qty or 0)
    return out


def worker_positions(farm) -> list[tuple[int, int]]:
    pos = [tuple(farm.get("farmer") or (4, 4))]
    for hand in farm.get("hands") or []:
        pos.append(tuple(hand))
    return pos


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(pos, dest) -> list:
    x, y = pos
    tx, ty = dest
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def shed_tiles(board: int = 10) -> list[tuple[int, int]]:
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def nearest_shed(pos, board: int = 10) -> tuple[int, int]:
    tiles = shed_tiles(board)
    return min(tiles, key=lambda t: manhattan(pos, t))
