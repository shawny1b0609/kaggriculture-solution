"""Node-select features: branch production plan, visible rival output, town demand.

Own logistics stay in ProgramVM. This vector is only for picking a forest child.
"""
from __future__ import annotations

from config import ANIMALS, CROPS, SELLABLE, SHOP_NAMES
from mining.tokens import parse_bundle

BAND_MID = {"0": 0, "1-2": 2, "3-5": 4, "6-9": 8, "10-14": 12, "15+": 16, "1-5": 3, "6+": 10}


def _mid(label: str) -> float:
    return float(BAND_MID.get(label or "0", 0))


def parse_token(token: str) -> dict:
    parts = (token or "").split("|")
    while len(parts) < 4:
        parts.append("?")
    crops = parse_bundle(parts[0])
    animals = parse_bundle(parts[1])
    workers = 1
    land = 1
    if parts[2].startswith("w"):
        try:
            workers = int(parts[2][1:] or 1)
        except ValueError:
            workers = 1
    if parts[3].startswith("l"):
        try:
            land = int(parts[3][1:] or 1)
        except ValueError:
            land = 1
    crop_mid = {name: _mid(crops.get(name[0], "0")) for name in CROPS}
    animal_mid = {name: _mid(animals.get(name[0], "0")) for name in ANIMALS}
    return {
        "crops": crops,
        "animals": animals,
        "crop_mid": crop_mid,
        "animal_mid": animal_mid,
        "workers": workers,
        "land": land,
    }


def _card_from_counts(crops: dict, animals: dict, land: int, program_id: str) -> dict:
    crops = {name: int(crops.get(name) or 0) for name in CROPS}
    animals = {name: int(animals.get(name) or 0) for name in ANIMALS}
    if sum(animals.values()) > 0:
        crops["WHEAT"] = max(int(crops.get("WHEAT") or 0), 8)
    return {
        "program_id": program_id,
        "crops": crops,
        "animals": animals,
        "hire_daily": 8,
        "land_target": max(1, min(4, int(land or 1))),
        "reserve_wheat": max(8, int(sum(animals.values()) * 2)),
        "open_animals": {name: min(2, n) for name, n in animals.items()},
        "shop_switch": {},
    }


def token_to_card(token: str, program_id: str = "forest_node") -> dict:
    parsed = parse_token(token)
    return _card_from_counts(parsed["crop_mid"], parsed["animal_mid"], parsed["land"], program_id)


def path_to_card(
    root_token: str,
    node_token: str,
    peek_token: str | None = None,
    node_day: int = 2,
    program_id: str = "forest_node",
) -> dict:
    """Card for one merged node. Optional peek is the immediate next child only.

    The forest merges identical prefixes. This card chases the occupancy of the
    node we are on, or the one-hop continuation those seats actually took next.
    It does not mix the opening of one game with a later unique segment of
    another, and it does not look down a unary spine several days ahead.
    `root_token` / `node_day` stay in the signature for callers; they do not
    splice extra occupancy in.
    """
    del root_token, node_day
    here = parse_token(node_token)
    nxt = parse_token(peek_token) if peek_token else here
    crops = {name: int(nxt["crop_mid"][name]) for name in CROPS}
    animals = {name: int(nxt["animal_mid"][name]) for name in ANIMALS}
    land = int(nxt["land"])
    return _card_from_counts(crops, animals, land, program_id)


def obs_feature_names() -> list[str]:
    names = ["day_frac"]
    names += [f"rival_{c.lower()}" for c in CROPS]
    names += [f"rival_{a.lower()}" for a in ANIMALS]
    names += [f"shop_{s}" for s in SHOP_NAMES]
    names += [f"demand_{p.lower()}" for p in SELLABLE]
    names += [f"demand_next_{p.lower()}" for p in SELLABLE]
    names += [f"price_{p.lower()}" for p in SELLABLE]
    return names


def branch_feature_names() -> list[str]:
    names = [f"plan_{c.lower()}" for c in CROPS]
    names += [f"plan_{a.lower()}" for a in ANIMALS]
    names += ["plan_land", "plan_workers"]
    return names


def _demand_vec(shops: list, step: int | None) -> tuple[list[float], list[float]]:
    from runtime.seller import shop_demand

    now = []
    nxt = []
    for item in SELLABLE:
        if step is None:
            from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS

            cap = 0.0
            for shop in shops or []:
                products = SHOPS.get(shop) or []
                if item in products:
                    cap += 2.0 if len(products) == 1 else 1.0
            now.append(min(cap, 8.0) / 8.0)
            nxt.append(min(cap, 8.0) / 8.0)
        else:
            now.append(min(shop_demand(shops, item, step), 8) / 8.0)
            nxt.append(min(shop_demand(shops, item, step + 1), 8) / 8.0)
    return now, nxt


def encode_obs(snap: dict, rival: dict | None = None, prices: dict | None = None, step: int | None = None) -> list[float]:
    shops = list(snap.get("shops") or [])
    shop_set = set(shops)
    vec = [min(int(snap.get("day") or 0), 29) / 29.0]
    rival = rival or {}
    rc = rival.get("crops") or {}
    ra = rival.get("animals") or {}
    for name in CROPS:
        vec.append(min(int(rc.get(name) or 0), 24) / 24.0)
    for name in ANIMALS:
        vec.append(min(int(ra.get(name) or 0), 16) / 16.0)
    vec += [1.0 if name in shop_set else 0.0 for name in SHOP_NAMES]
    now, nxt = _demand_vec(shops, step if step is not None else snap.get("step"))
    vec += now
    vec += nxt
    prices = prices or {}
    for item in SELLABLE:
        vec.append(min(float(prices.get(item) or 0), 400.0) / 400.0)
    return vec


def encode_obs_from_env(obs: dict) -> list[float]:
    from runtime.view import counts, rival_farm, step_of

    rival = counts(rival_farm(obs))
    market = obs.get("market") or {}
    prices = dict(market.get("prices") or {})
    snap = {
        "day": obs.get("day") or 0,
        "shops": list((obs.get("town") or {}).get("unlocked_shops") or []),
        "step": step_of(obs),
    }
    return encode_obs(snap, {"crops": rival["crops"], "animals": rival["animals"]}, prices, step=step_of(obs))


def encode_branch(parent_token: str, child_token: str, child_node: dict | None = None) -> list[float]:
    child = parse_token(child_token)
    vec = [child["crop_mid"][name] / 16.0 for name in CROPS]
    vec += [child["animal_mid"][name] / 16.0 for name in ANIMALS]
    vec.append(min(child["land"], 4) / 4.0)
    vec.append(min(child["workers"], 12) / 12.0)
    return vec


def encode_dest_branch(dest: dict | None) -> list[float]:
    """Exact crop/animal counts of a chosen next grid. Same layout as encode_branch."""
    dest = dest or {}
    crops = dest.get("crops") or {}
    animals = dest.get("animals") or {}
    vec = [min(int(crops.get(name) or 0), 24) / 24.0 for name in CROPS]
    vec += [min(int(animals.get(name) or 0), 16) / 16.0 for name in ANIMALS]
    vec.append(min(int(dest.get("land") or 1), 4) / 4.0)
    vec.append(8.0 / 12.0)
    return vec


OBS_DIM = len(obs_feature_names())
BRANCH_DIM = len(branch_feature_names())
MAX_CHOICES = 32
