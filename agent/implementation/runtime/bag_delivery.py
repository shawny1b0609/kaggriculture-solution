"""Advance a shed-side product deposit without moving any production event."""
from __future__ import annotations

import copy

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from config import SELLABLE
from runtime.weed_repair import MOVES


def _actions(step: dict) -> list[list]:
    return [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]


def _write_action(step: dict, worker: int, action: list) -> None:
    if worker:
        step["hands"][worker - 1] = action
    else:
        step["farmer"] = action


def _can_delay(tape: list[dict], worker: int, start: int, end: int) -> bool:
    for hour in range(start, end):
        step = tape[hour]
        action = _actions(step)[worker]
        if action[0] in ("HARVEST", "COLLECT_FERTILIZER", "PICKUP", "DROP", "PLACE", "DIG"):
            return False
        positions = step.get("positions") or []
        next_positions = tape[hour + 1].get("positions") or []
        if worker >= len(positions) or worker >= len(next_positions):
            return False
        if int(step.get("post_hands") or 0) > len(positions) - 1:
            return False
        if action[0] in MOVES or action == ["PASS"]:
            continue
        position = positions[worker]
        for check_hour in (hour, hour + 1):
            for other, other_action in enumerate(_actions(tape[check_hour])):
                if other == worker or other_action[0] in MOVES or other_action == ["PASS"]:
                    continue
                other_positions = tape[check_hour].get("positions") or []
                if other < len(other_positions) and other_positions[other] == position:
                    return False
    return True


def plan_early_deposit(obs: dict, tape: list[dict], hour: int) -> dict | None:
    farm = obs["farms"][obs["player"]]
    private = obs.get("private") or {}
    bags = private.get("inventories") or []
    positions = [farm["farmer"], *(farm.get("hands") or [])]
    room = 100 - sum(int(amount or 0) for amount in (private.get("shed") or {}).values())
    if hour >= len(tape) or room <= 0:
        return None
    prices = (obs.get("market") or {}).get("prices") or {}
    for worker, action in enumerate(_actions(tape[hour])):
        if (worker >= len(bags) or worker >= len(positions)
                or action[0] not in ("PASS", *MOVES)
                or not engine._is_shed_adjacent(tuple(positions[worker]), len(farm["tiles"]))):
            continue
        expected = tape[hour].get("positions") or []
        if worker >= len(expected) or positions[worker] != expected[worker]:
            continue
        future_actions = [_actions(step)[worker] for step in tape[hour:]
                          if worker < len(_actions(step))]
        eligible = []
        for item, amount in (bags[worker] or {}).items():
            if item not in SELLABLE or int(amount) <= 0 or int(amount) > room:
                continue
            consume = "FEED" if item == "WHEAT" else "FERTILIZE" if item == "FERTILIZER" else None
            if consume and any(later[0] == consume for later in future_actions):
                continue
            eligible.append((int(amount) * float(prices.get(item) or 1), item, int(amount)))
        if not eligible:
            continue
        _, item, quantity = max(eligible)
        stop = next((index for index in range(hour, len(tape))
                     if worker < len(_actions(tape[index])) and _actions(tape[index])[worker] == ["PASS"]), None)
        terminal = False
        if stop is None:
            stop = len(tape) - 1
            final_action = _actions(tape[stop])[worker]
            terminal = final_action[0] in MOVES
            if not terminal:
                continue
        if not _can_delay(tape, worker, hour, stop):
            continue
        if int(tape[stop].get("post_hands") or 0) > len(tape[stop].get("positions") or []) - 1:
            continue
        originals = [copy.deepcopy(tape[index]) for index in range(hour, stop + 1)]
        position = list(positions[worker])
        for offset, index in enumerate(range(hour, stop + 1)):
            step = tape[index]
            step["positions"][worker] = list(position)
            shifted = ["PLACE", item, quantity] if offset == 0 else list(_actions(originals[offset - 1])[worker])
            _write_action(step, worker, shifted)
            events = step.get("unit_events")
            if events is not None and worker < len(events):
                events[worker] = ({"seeds": {}, "shed": {item: quantity}} if offset == 0
                                  else copy.deepcopy(originals[offset - 1]["unit_events"][worker]))
            movement = MOVES.get(shifted[0], (0, 0))
            position = [position[0] + movement[0], position[1] + movement[1]]
        remaining_return = quantity
        for step in tape[stop + 1:]:
            events = step.get("unit_events") or []
            if worker >= len(events):
                continue
            returns = events[worker].get("shed") or {}
            reduction = min(remaining_return, max(0, int(returns.get(item) or 0)))
            if reduction:
                returns[item] -= reduction
                remaining_return -= reduction
        for step in tape[hour:]:
            if "unit_events" not in step:
                continue
            for bucket, aggregate in (("shed", "shed_delta_units"), ("seeds", "seed_delta_units")):
                totals = {}
                for event in step["unit_events"]:
                    for resource, delta in (event.get(bucket) or {}).items():
                        totals[resource] = totals.get(resource, 0) + int(delta)
                step[aggregate] = {resource: amount for resource, amount in totals.items() if amount}
        return {"day": int(obs.get("day") or 0), "hour": hour, "worker": worker,
                "item": item, "quantity": quantity, "absorbed_at": stop,
                "discarded_terminal_move": terminal}
    return None
