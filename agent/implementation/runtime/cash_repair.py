"""Local field repair when a seed purchase did not fill.

The market follows the field phase.  A failed purchase cannot finance today's
current-hour PLANT, so move only that worker's local, non-yielding suffix into
an existing PASS (or discard an irrelevant final move).  Never assume a future
purchase will fill: the live seed count is checked again every hour.
"""
from __future__ import annotations

import copy

from runtime.weed_repair import (_insert_worker_delay, _recompute_step_events,
                                 _replace_idle_action)


def _units(step: dict) -> list[list]:
    return [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]


def _plant_recovered_before_midnight(tape: list[dict], hour: int, xy: tuple[int, int],
                                     crop: str) -> bool:
    planted = None
    for later in range(hour + 1, len(tape)):
        step = tape[later]
        for worker, action in enumerate(_units(step)):
            positions = step.get("positions") or []
            if worker >= len(positions) or tuple(positions[worker]) != xy or not action:
                continue
            when = (later, worker)
            if action == ["PLANT", crop]:
                planted = when
            elif planted is None and action[0] in ("FERTILIZE", "HARVEST", "PLACE", "BUILD_COOP", "BUILD_PASTURE"):
                return False
            elif planted is not None and action == ["WATER"] and when > planted:
                return True
            elif planted is not None and action[0] in ("DIG", "HARVEST"):
                return False
    return False


def postpone_unfunded_plants(farm: dict, private: dict, tape: list[dict],
                             hour: int,
                             protected_slots: set[tuple[int, int]] | None = None,
                             ) -> tuple[list[dict], list[dict]]:
    """Retain only currently affordable PLANTs; reschedule others within today.

    A same-day reschedule needs a later WATER.  Unrecoverable PLANTs are
    returned per tile for cross-day local recovery. Same-hour PLANT requests
    are atomic by crop: if the group still exceeds stock, *all* its remaining
    requests fail, not just the excess requests.
    """
    if hour >= len(tape):
        return [], []
    positions = tape[hour].get("positions") or []
    actual = [farm.get("farmer") or [], *(farm.get("hands") or [])]
    seeds = private.get("seeds") or {}
    protected_slots = protected_slots or set()
    repairs, blocked = [], []
    crops = sorted({str(action[1]) for action in _units(tape[hour])
                    if len(action) >= 2 and action[0] == "PLANT"})
    for crop in crops:
        demand = [(worker, tuple(positions[worker]))
                  for worker, action in enumerate(_units(tape[hour]))
                  if len(action) >= 2 and action == ["PLANT", crop]
                  and worker < len(positions)]
        shortage = max(0, len(demand) - int(seeds.get(crop) or 0))
        if not shortage:
            continue
        # Defer later-index workers first; earlier source workers retain the
        # scarce current-hour seeds and the engine's original execution order.
        for worker, xy in reversed(demand):
            if shortage <= 0:
                break
            if worker >= len(actual) or tuple(actual[worker]) != xy:
                continue
            x, y = xy
            tile = farm["tiles"][y][x]
            if tile is not None:
                continue
            trial = copy.deepcopy(tape)
            detail = {}
            mode = None
            if _insert_worker_delay(
                trial, worker, hour, xy, ["PASS"], detail=detail,
                protected_slots=protected_slots, allow_terminal_move=True,
            ) and _plant_recovered_before_midnight(trial, hour, xy, crop):
                mode = "shift_worker_into_slack"
            else:
                # A different worker already standing here may absorb the
                # PLANT in a later PASS; the original worker is unaffected.
                trial = copy.deepcopy(tape)
                for later in range(hour + 1, len(trial)):
                    for other, pos in enumerate(trial[later].get("positions") or []):
                        if (tuple(pos) != xy or other >= len(actual)
                                or tuple(actual[other]) != xy
                                or (later, other) in protected_slots
                                or _units(trial[later])[other] != ["PASS"]):
                            continue
                        if any(
                            other >= len(trial[middle].get("positions") or [])
                            or tuple(trial[middle]["positions"][other]) != xy
                            or _units(trial[middle])[other] != ["PASS"]
                            or (middle, other) in protected_slots
                            for middle in range(hour, later)
                        ):
                            continue
                        if worker == 0:
                            trial[hour]["farmer"] = ["PASS"]
                        else:
                            trial[hour]["hands"][worker - 1] = ["PASS"]
                        events = trial[hour].get("unit_events")
                        if events is not None and worker < len(events):
                            events[worker] = {"seeds": {}, "shed": {}}
                            _recompute_step_events(trial[hour])
                        _replace_idle_action(trial[later], other, ["PLANT", crop])
                        if _plant_recovered_before_midnight(trial, hour, xy, crop):
                            mode = "handoff_to_idle_worker"
                            detail = {"replacement_hour": later, "replacement_worker": other}
                            break
                        trial = copy.deepcopy(tape)
                    if mode:
                        break
            if mode:
                tape[:] = trial
                shortage -= 1
                repairs.append({"hour": hour, "worker": worker, "crop": crop,
                                "xy": list(xy), "mode": mode, **detail})
        if shortage:
            # The engine cancels the entire crop group.  Record every lost
            # tile, and remove phantom consumption from the tape so the
            # market planner does not reserve seed for a known failed action.
            remaining = [(worker, tuple(positions[worker]))
                         for worker, action in enumerate(_units(tape[hour]))
                         if action == ["PLANT", crop] and worker < len(positions)]
            for worker, xy in remaining:
                _replace_idle_action(tape[hour], worker, ["PASS"])
                blocked.append({"hour": hour, "worker": worker, "xy": list(xy),
                                "crop": crop, "missing": shortage,
                                "reason": "no_same_day_plant_and_water_slot"})
    return repairs, blocked


def handoff_missed_maintenance(farm: dict, private: dict, step: dict,
                               farmer: list, hands: list[list],
                               ) -> tuple[list, list[list], list[dict]]:
    """Give a missed WATER/FEED/CARE to an already co-located idle worker.

    The substitute must already be on the tile; no source movement or hire
    order changes.  FEED also requires wheat in *that worker's* personal bag.
    The caller enables this only after an actual market staffing/input miss.
    """
    actions = [list(farmer), *[list(action) for action in hands]]
    source_positions = step.get("positions") or []
    actual_positions = [farm.get("farmer") or [], *(farm.get("hands") or [])]
    bags = private.get("inventories") or []
    repairs = []
    for worker, action in enumerate(list(actions)):
        if not action or action[0] not in ("WATER", "FEED", "CARE"):
            continue
        if worker >= len(source_positions):
            continue
        xy = tuple(source_positions[worker])
        x, y = xy
        tile = farm["tiles"][y][x]
        op = action[0]
        if op == "WATER" and not (isinstance(tile, dict) and tile.get("kind") == "PLANT"
                                    and not tile.get("watered_today")):
            continue
        if op in ("FEED", "CARE") and not (isinstance(tile, dict) and tile.get("animal")
                                             and not tile.get("fed_today" if op == "FEED" else "cared_today")):
            continue
        in_place = worker < len(actual_positions) and tuple(actual_positions[worker]) == xy
        stocked = op != "FEED" or (worker < len(bags) and int((bags[worker] or {}).get("WHEAT") or 0) > 0)
        if in_place and stocked:
            continue
        for helper, position in enumerate(actual_positions):
            if (helper == worker or helper >= len(actions) or tuple(position) != xy
                    or actions[helper] != ["PASS"]):
                continue
            if op == "FEED" and (helper >= len(bags)
                                 or int((bags[helper] or {}).get("WHEAT") or 0) <= 0):
                continue
            actions[helper] = list(action)
            actions[worker] = ["PASS"]
            repairs.append({"op": op, "xy": list(xy), "source_worker": worker,
                            "helper": helper})
            break
    return actions[0], actions[1:], repairs
