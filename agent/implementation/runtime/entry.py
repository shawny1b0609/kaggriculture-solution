"""Forest agent: equal opening prior, later weight by native win-rate. Not shop-name tape switch."""
from __future__ import annotations

import hashlib
import random

from runtime.program_vm import ProgramVM, default_programs
from runtime.view import counts, own_farm, rival_farm, step_of


def equal_weights(program_ids: list[str]) -> dict[str, float]:
    return {pid: 1.0 for pid in program_ids}


def sample_program(weights: dict[str, float], rng: random.Random) -> str:
    items = [(pid, max(1e-9, float(w))) for pid, w in weights.items()]
    total = sum(w for _, w in items)
    cut = rng.random() * total
    acc = 0.0
    for pid, w in items:
        acc += w
        if acc >= cut:
            return pid
    return items[-1][0]


def update_weights(weights: dict[str, float], chosen: str, win_score: float, eta: float = 0.25) -> dict[str, float]:
    """Multiplicative weights. win_score in {0, 0.5, 1}."""
    out = dict(weights)
    out[chosen] = float(out.get(chosen) or 1.0) * (1.0 + eta * (float(win_score) - 0.5))
    return out


def _prices(obs) -> dict:
    market = obs.get("market") or {}
    return dict(market.get("prices") or {})


def program_score(pid: str, card: dict, obs, shops: list[str], own, rival, weight: float) -> float:
    """Observation-conditioned score. Shops, visible rival herd, market prices, own state."""
    score = max(1e-9, float(weight))
    shop_set = set(shops)
    prices = _prices(obs)
    wool = float(prices.get("WOOL") or 200)
    milk = float(prices.get("MILK") or 160)
    egg = float(prices.get("EGG") or 50)
    berry = float(prices.get("STRAWBERRY") or 120)
    melon = float(prices.get("MELON") or 250)
    want_sheep = int(card.get("animals", {}).get("SHEEP") or 0)
    want_cow = int(card.get("animals", {}).get("COW") or 0)
    want_goose = int(card.get("animals", {}).get("GOOSE") or 0)
    want_berry = int(card.get("crops", {}).get("STRAWBERRY") or 0)
    want_melon = int(card.get("crops", {}).get("MELON") or 0)
    rival_sheep = int(rival["animals"].get("SHEEP") or 0)
    rival_cow = int(rival["animals"].get("COW") or 0)
    own_animals = sum(int(v or 0) for v in own["animals"].values())
    dest_animals = want_sheep + want_cow + want_goose
    if own_animals >= 6 and dest_animals == 0:
        return 0.0
    if "YARN_STORE" in shop_set:
        score *= 1.25 if want_sheep >= 6 and wool >= 40 else (0.55 if want_sheep >= 6 else 1.0)
    if rival_sheep >= 8 and want_sheep >= 6:
        score *= 0.65
    if "PIZZA_SHOP" in shop_set or "ICE_CREAM_SHOP" in shop_set:
        score *= 1.2 if want_cow >= 6 and milk >= 80 else 1.0
    if "SMOOTHIE_SHOP" in shop_set or "BRUNCH_SPOT" in shop_set:
        score *= 1.15 if want_berry >= 8 and berry >= 80 else 1.0
    if "PET_CAFE" in shop_set:
        score *= 1.1 if int(card.get("crops", {}).get("CARROT") or 0) >= 8 else 0.95
    if rival_cow >= 8 and want_cow >= 8 and milk < 80:
        score *= 0.7
    if want_goose >= 6 and egg < 20:
        score *= 0.6
    if want_melon >= 10 and melon < 40:
        score *= 0.55
    if own["land"] < int(card.get("land_target") or 1) and own["cash"] < 800:
        score *= 0.9
    return score


class ForestAgent:
    """mode: weighted | best_single | equal_open_only."""

    def __init__(self, programs: dict | None = None, weights: dict | None = None, mode: str = "weighted", seed: int | None = None):
        self.library = programs or default_programs()
        self.weights = dict(weights or equal_weights(list(self.library)))
        self.mode = mode
        self.rng = random.Random(seed if seed is not None else 0)
        if mode == "best_single":
            self.active_id = max(self.weights, key=lambda k: (self.weights[k], k))
            self.locked = True
        else:
            self.active_id = sample_program(self.weights, self.rng)
            self.locked = False
        self.vm = ProgramVM(self.library[self.active_id])
        self.calls = 0
        self.program_id = self.active_id
        self.choice = {"root": self.active_id, "mode": mode, "switches": []}
        self.prev_shops = []
        self.prev_day = -1

    def __call__(self, obs, cfg=None):
        self.calls += 1
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        day = int(obs.get("day") or 0)
        hour = int(obs.get("hour") or 0)
        if not self.locked and (shops != self.prev_shops or (hour == 0 and day != self.prev_day)):
            self._maybe_switch(obs, shops)
        self.prev_shops = shops
        self.prev_day = day
        action = self.vm(obs, cfg)
        self.program_id = self.active_id
        return action

    def _maybe_switch(self, obs, shops: list[str]):
        farm = own_farm(obs)
        own = counts(farm)
        rival = counts(rival_farm(obs))
        scored = []
        for pid, card in self.library.items():
            scored.append((program_score(pid, card, obs, shops, own, rival, self.weights.get(pid, 1.0)), pid))
        scored.sort(reverse=True)
        best_score, target = scored[0]
        cur = program_score(self.active_id, self.library[self.active_id], obs, shops, own, rival, self.weights.get(self.active_id, 1.0))
        if target == self.active_id or best_score < cur * 1.08:
            return
        dest = self.library[target]
        cur_animals = sum(own["animals"].values())
        dest_animals = sum(int(v or 0) for v in dest["animals"].values())
        if cur_animals >= 6 and dest_animals == 0:
            return
        self.choice["switches"].append({
            "step": step_of(obs),
            "from": self.active_id,
            "to": target,
            "shops": shops,
            "rival_animals": rival["animals"],
            "prices": {k: _prices(obs).get(k) for k in ("WOOL", "MILK", "EGG", "STRAWBERRY", "MELON")},
        })
        self.active_id = target
        self.vm = ProgramVM(self.library[target])
        self.program_id = target


def forest_factory(mode: str, weights: dict | None = None, programs: dict | None = None):
    def factory():
        seed = int(hashlib.sha256(f"{mode}-{id(object())}".encode()).hexdigest()[:8], 16)
        return ForestAgent(programs=programs, weights=weights, mode=mode, seed=seed)

    return factory
