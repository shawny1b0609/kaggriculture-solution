"""Remove source worker instructions proven inert for this day's production."""
from __future__ import annotations

from config import ANIMALS


def trim_consumable_pickups(tape: list[dict]) -> list[dict]:
    """Back-propagate each worker's actual feed/fertilize consumption.

    Drop-surplus is not a production obligation. After shortening a pickup,
    shorten its corresponding source shed-return delta too; the planner will
    provision any other worker's needs directly from the communal shed.
    Harvest/collection income is conservatively ignored, so this is an upper
    bound on required pickups, not a speculative minimum.
    """
    workers = max((1 + len(step.get("hands") or []) for step in tape), default=0)
    cuts: dict[tuple[int, int, str], int] = {}
    for worker in range(workers):
        for item, consume in (("WHEAT", "FEED"), ("FERTILIZER", "FERTILIZE")):
            need = 0
            for hour in range(len(tape) - 1, -1, -1):
                step = tape[hour]
                units = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
                if worker >= len(units):
                    continue
                action = units[worker]
                if not isinstance(action, list) or not action:
                    continue
                if action[0] == "DROP":
                    need = 0
                elif action[0] == consume:
                    need += 1
                elif action[0] == "PICKUP" and len(action) >= 3 and action[1] == item:
                    events = step.get("unit_events") or []
                    taken = -int((events[worker].get("shed") or {}).get(item) or 0) if worker < len(events) else 0
                    requested = int(action[2] or 0)
                    if taken <= 0 or taken != requested:
                        continue
                    keep = min(taken, need)
                    need -= keep
                    if keep < taken:
                        cuts[(hour, worker, item)] = taken - keep
    removed = []
    bag_reductions: dict[tuple[int, str], int] = {}
    for hour, step in enumerate(tape):
        units = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
        events = step.get("unit_events") or []
        for worker, action in enumerate(units):
            if worker >= len(events):
                continue
            delta = events[worker].setdefault("shed", {})
            for item in ("WHEAT", "FERTILIZER"):
                key = (worker, item)
                reduction = cuts.get((hour, worker, item), 0)
                if reduction:
                    keep = int(action[2]) - reduction
                    new_action = ["PICKUP", item, keep] if keep else ["PASS"]
                    if worker == 0:
                        step["farmer"] = new_action
                    else:
                        step["hands"][worker - 1] = new_action
                    delta[item] = int(delta.get(item) or 0) + reduction
                    if not delta[item]:
                        delta.pop(item)
                    bag_reductions[key] = bag_reductions.get(key, 0) + reduction
                    removed.append({"hour": hour, "worker": worker, "item": item,
                                    "source_effective": int(action[2]), "kept": keep})
                returned = max(0, int(delta.get(item) or 0))
                if returned and bag_reductions.get(key, 0):
                    less = min(returned, bag_reductions[key])
                    delta[item] -= less
                    if not delta[item]:
                        delta.pop(item)
                    bag_reductions[key] -= less
                if action == ["DROP"]:
                    bag_reductions[key] = 0  # overflow, if any, is discarded too
        if events:
            total = {}
            for event in events:
                for item, amount in (event.get("shed") or {}).items():
                    total[item] = total.get(item, 0) + int(amount)
            step["shed_delta_units"] = {item: n for item, n in total.items() if n}
    return removed


def remove_dead_animal_pickups(tape: list[dict]) -> list[dict]:
    """Omit animal pickups never used by that worker before the day reset.

    Animal items can only change the farm grid through a later PLACE by the
    same worker. If that never happens, the official end-of-day bag drop puts
    the animal back in the shed. Do not alter any pickup with a later PLACE,
    nor any other item or productive action.
    """
    removed = []
    for hour, step in enumerate(tape):
        units = [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]
        for worker, action in enumerate(units):
            if not isinstance(action, list) or len(action) < 2 or action[0] != "PICKUP":
                continue
            item = str(action[1])
            if item not in ANIMALS:
                continue
            used_later = False
            for later in tape[hour + 1:]:
                later_units = [later.get("farmer") or ["PASS"], *(later.get("hands") or [])]
                if (worker < len(later_units) and isinstance(later_units[worker], list)
                        and len(later_units[worker]) >= 2
                        and later_units[worker][0] == "PLACE" and later_units[worker][1] == item):
                    used_later = True
                    break
            if used_later:
                continue
            # A DROP transfers the animal back to communal stock for another
            # worker, and a later PICKUP of the same item may rely on that
            # transfer. Leave those chains untouched; only terminal carrying
            # with no same-day redeposit is certified here.
            redeposited = False
            for later in tape[hour + 1:]:
                later_units = [later.get("farmer") or ["PASS"], *(later.get("hands") or [])]
                if worker < len(later_units) and later_units[worker] == ["DROP"]:
                    redeposited = True
                    break
            if redeposited:
                continue
            if worker == 0:
                step["farmer"] = ["PASS"]
            else:
                step["hands"][worker - 1] = ["PASS"]
            events = step.get("unit_events") or []
            released = 0
            if worker < len(events):
                released = -min(0, int((events[worker].get("shed") or {}).pop(item, 0)))
            if released:
                totals = step.get("shed_delta_units") or {}
                totals[item] = int(totals.get(item) or 0) + released
                if totals[item] == 0:
                    totals.pop(item)
                step["shed_delta_units"] = totals
            removed.append({"hour": hour, "worker": worker, "item": item,
                            "source_request": int(action[2]) if len(action) >= 3 else 1,
                            "source_effective": released})
    return removed
