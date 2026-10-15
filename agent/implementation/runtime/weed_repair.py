"""Repair dawn-visible weeds while preserving the source production clock.

For every weed that blocks a later PLANT/BUILD, the preferred repair is the
first source worker that reaches the tile and can absorb one extra DIG in a
later PASS.  Candidates are tried in official execution order, including the
worker that was supposed to perform the blocked job.  A dedicated hired hand
is planned by the caller only after every source-worker opportunity fails.
"""
from __future__ import annotations

import copy

from mining.grid_key import is_weed

EMPTY_REQUIRED = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "WEST": (-1, 0), "EAST": (1, 0)}


def _units(step: dict) -> list[list]:
    return [step.get("farmer") or ["PASS"], *(step.get("hands") or [])]


def _weed_needs(farm: dict, tape: list[dict]) -> dict[tuple[int, int], tuple[int, int]]:
    """Earliest source operation that requires each dawn weed to be gone."""
    tiles = farm.get("tiles") or []
    needs: dict[tuple[int, int], tuple[int, int]] = {}
    for hour, step in enumerate(tape):
        positions = step.get("positions") or []
        for idx, action in enumerate(_units(step)):
            if not isinstance(action, list) or not action or action[0] not in EMPTY_REQUIRED or idx >= len(positions):
                continue
            x, y = positions[idx]
            if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y]) and is_weed(tiles[y][x])):
                continue
            xy = (int(x), int(y))
            at = (hour, idx)
            if xy not in needs or at < needs[xy]:
                needs[xy] = at
    return needs


def _has_source_dig(tape: list[dict], xy: tuple[int, int], deadline: tuple[int, int]) -> bool:
    for hour, step in enumerate(tape):
        positions = step.get("positions") or []
        for idx, action in enumerate(_units(step)):
            if ((hour, idx) < deadline and idx < len(positions)
                    and tuple(positions[idx]) == xy and action == ["DIG"]):
                return True
    return False


def plan_idle_digs(farm: dict, tape: list[dict]) -> tuple[dict[tuple[int, int], tuple[int, int]], list[dict]]:
    """Return {(hour, worker): (x,y)} and targets without a free local slot."""
    needs = _weed_needs(farm, tape)
    plan: dict[tuple[int, int], tuple[int, int]] = {}
    blocked = []
    for xy, deadline in needs.items():
        already_digged = False
        slots = []
        for hour, step in enumerate(tape):
            positions = step.get("positions") or []
            for idx, action in enumerate(_units(step)):
                when = (hour, idx)
                if when >= deadline or idx >= len(positions) or tuple(positions[idx]) != xy:
                    continue
                if isinstance(action, list) and action and action[0] == "DIG":
                    already_digged = True
                elif isinstance(action, list) and action and action[0] == "PASS" and when not in plan:
                    slots.append(when)
        if already_digged:
            continue
        if slots:
            # Clear at the first free opportunity.  Waiting for the latest
            # possible PASS made a small perturbation needlessly fragile.
            plan[min(slots)] = xy
        else:
            blocked.append({"xy": list(xy), "deadline_hour": deadline[0], "worker": deadline[1], "reason": "no_local_idle_dig_slot"})
    return plan, blocked


def apply_idle_digs(farm: dict, hour: int, farmer: list, hands: list[list], plan: dict) -> tuple[list, list[list], bool]:
    units = [list(farmer), *[list(hand) for hand in hands]]
    positions = [farm.get("farmer") or [], *(farm.get("hands") or [])]
    used = False
    for (planned_hour, idx), xy in plan.items():
        if planned_hour != hour or idx >= len(units) or idx >= len(positions):
            continue
        x, y = xy
        if tuple(positions[idx]) != xy or units[idx][0] != "PASS":
            continue
        tile = farm["tiles"][y][x]
        if is_weed(tile):
            units[idx] = ["DIG"]
            used = True
    return units[0], units[1:], used


def _replace_unit(step: dict, idx: int, action: list) -> None:
    if idx == 0:
        step["farmer"] = action
    else:
        step["hands"][idx - 1] = action


def repair_same_hour_plant_water(tape: list[dict]) -> int:
    """Reassign co-located PLANT/WATER so the official worker order is legal.

    A weed delay can move PLANT into the source WATER hour. Both workers stand
    on the same tile, and seeds are global, so swapping their *non-yield* jobs
    preserves the source day's product curve and their later positions.
    """
    swaps = 0
    for step in tape:
        positions = step.get("positions") or []
        units = _units(step)
        for plant_idx, plant in enumerate(units):
            if not plant or plant[0] != "PLANT" or plant_idx >= len(positions):
                continue
            water_idx = next((i for i in range(plant_idx)
                              if i < len(positions) and tuple(positions[i]) == tuple(positions[plant_idx])
                              and units[i] == ["WATER"]), None)
            if water_idx is None:
                continue
            _replace_unit(step, water_idx, list(plant))
            _replace_unit(step, plant_idx, ["WATER"])
            events = step.get("unit_events")
            if events is not None and water_idx < len(events) and plant_idx < len(events):
                events[water_idx], events[plant_idx] = events[plant_idx], events[water_idx]
            units[water_idx], units[plant_idx] = units[plant_idx], units[water_idx]
            swaps += 1
    return swaps


def repair_same_hour_field_dependencies(tape: list[dict]) -> list[dict]:
    """Restore field-operation ordering broken by a weed-induced delay.

    Source tapes may deliberately perform two stationary jobs on one tile in
    the same hour.  They are legal only because the lower worker index runs
    first.  Inserting a DIG earlier in the day can move a prerequisite to a
    later worker index without moving the dependent action, for example::

        BUILD_PASTURE (worker 0), DIG (worker 5)
        PLANT (worker 2), HARVEST (worker 4)

    The first case is repaired by swapping the bag-independent DIG and
    PLANT/BUILD actions.  The second keeps HARVEST with its original worker so
    the product remains in the correct bag, and delays the dependent action
    into that worker's later PASS.  The operation is transactional through
    ``_insert_worker_delay``; protected yield and warehouse actions are never
    shifted merely to force a repair.
    """
    repaired: list[dict] = []
    for _ in range(40):
        changed = False
        for hour, step in enumerate(tape):
            positions = step.get("positions") or []
            units = _units(step)
            for dependent_idx, dependent in enumerate(units):
                if dependent_idx >= len(positions) or not dependent:
                    continue
                dep_op = dependent[0]
                if dep_op not in EMPTY_REQUIRED and dep_op != "DIG":
                    continue
                xy = tuple(positions[dependent_idx])
                prerequisite = next(
                    (
                        (idx, action)
                        for idx, action in enumerate(units)
                        if idx > dependent_idx
                        and idx < len(positions)
                        and tuple(positions[idx]) == xy
                        and action
                        and (
                            (action[0] == "DIG" and dep_op in EMPTY_REQUIRED)
                            or (action[0] == "HARVEST" and dep_op in EMPTY_REQUIRED | {"DIG"})
                        )
                    ),
                    None,
                )
                if prerequisite is None:
                    continue
                prerequisite_idx, prerequisite_action = prerequisite
                if prerequisite_action[0] == "DIG":
                    # DIG and PLANT/BUILD are stationary and carry no bag
                    # ownership.  Swapping them restores official unit order
                    # without changing either worker's later position.
                    _replace_unit(step, dependent_idx, list(prerequisite_action))
                    _replace_unit(step, prerequisite_idx, list(dependent))
                    events = step.get("unit_events")
                    if (events is not None and dependent_idx < len(events)
                            and prerequisite_idx < len(events)):
                        events[dependent_idx], events[prerequisite_idx] = (
                            events[prerequisite_idx], events[dependent_idx]
                        )
                        _recompute_step_events(step)
                    repaired.append({
                        "hour": hour, "xy": list(xy), "mode": "swap_dig_before_field_job",
                        "prerequisite_worker": prerequisite_idx,
                        "dependent_worker": dependent_idx,
                        "dependent_action": list(dependent),
                    })
                    changed = True
                    break
                # HARVEST must stay with its source worker because output goes
                # into that worker's bag.  Wait one hour before the dependent
                # DIG/PLANT/BUILD and absorb the delay in a later PASS.
                if _insert_worker_delay(
                    tape, dependent_idx, hour, xy, ["PASS"]
                ):
                    repaired.append({
                        "hour": hour, "xy": list(xy), "mode": "delay_until_after_harvest",
                        "prerequisite_worker": prerequisite_idx,
                        "dependent_worker": dependent_idx,
                        "dependent_action": list(dependent),
                    })
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return repaired


def _route_for_one_dig(tape: list[dict], worker: int, xy: tuple[int, int], deadline: tuple[int, int], board_size: int):
    first = next((h for h, step in enumerate(tape) if worker < len(step.get("positions") or [])), None)
    if first is None or (first, worker) >= deadline:
        return None
    start = tuple(tape[first]["positions"][worker])
    # State: x, y, has_cleared_target. All fixed worker jobs retain their
    # original hour and position; only moves and PASS are re-routed.
    layer = {(start[0], start[1], False): 0}
    parents = []
    for hour in range(first, len(tape)):
        step = tape[hour]
        original = _units(step)[worker]
        source_pos = tuple(step["positions"][worker])
        op = original[0] if original else "PASS"
        fixed = op not in MOVES and op != "PASS"
        next_pos = None
        if hour + 1 < len(tape) and worker < len(tape[hour + 1].get("positions") or []):
            next_pos = tuple(tape[hour + 1]["positions"][worker])
        hire_checkpoint = any(order and order[0] == "HIRE" for order in step.get("market") or [])
        following = {}
        links = {}
        for state, cost in layer.items():
            x, y, cleared = state
            if fixed and (x, y) != source_pos:
                continue
            choices = [op] if fixed else ["PASS", *MOVES]
            if not fixed and not cleared and (x, y) == xy and (hour, worker) < deadline:
                choices.append("DIG")
            for chosen in choices:
                dx, dy = MOVES.get(chosen, (0, 0))
                nx, ny = x + dx, y + dy
                if not (0 <= nx < board_size and 0 <= ny < board_size):
                    continue
                if hire_checkpoint and next_pos is not None and (nx, ny) != next_pos:
                    continue
                # A fixed source DIG on another tile is not a repair for this
                # weed. It previously set the cleared flag anywhere on the
                # board, causing false route candidates and 20 retry loops.
                done = cleared or (chosen == "DIG" and (x, y) == xy and (hour, worker) < deadline)
                new_state = (nx, ny, done)
                step_cost = 0 if chosen == op else 2
                if next_pos is not None and (nx, ny) != next_pos:
                    step_cost += 1
                total = cost + step_cost
                if new_state not in following or total < following[new_state]:
                    following[new_state] = total
                    links[new_state] = (state, chosen, (x, y))
        if not following:
            return None
        layer = following
        parents.append(links)
    completed = [(cost, state) for state, cost in layer.items() if state[2]]
    if not completed:
        return None
    cost, state = min(completed)
    changes = []
    for offset in range(len(parents) - 1, -1, -1):
        prev, chosen, before = parents[offset][state]
        changes.append((first + offset, before, chosen))
        state = prev
    changes.reverse()
    return cost, changes


def repair_routes(farm: dict, tape: list[dict]) -> tuple[dict, list[dict], int]:
    """Re-route idle/move hours around fixed productive jobs to insert DIG."""
    routes_changed = 0
    for _ in range(20):
        local_plan, blocked = plan_idle_digs(farm, tape)
        if not blocked:
            return local_plan, [], routes_changed
        target = blocked[0]
        xy = tuple(target["xy"])
        deadline = (int(target["deadline_hour"]), int(target["worker"]))
        workers = max((len(step.get("positions") or []) for step in tape), default=0)
        candidates = []
        for worker in range(workers):
            candidate = _route_for_one_dig(tape, worker, xy, deadline, len(farm.get("tiles") or []))
            if candidate is not None:
                candidates.append((candidate[0], worker, candidate[1]))
        if not candidates:
            return local_plan, blocked, routes_changed
        _, worker, changes = min(candidates, key=lambda row: (row[0], row[1]))
        for hour, before, action in changes:
            tape[hour]["positions"][worker] = list(before)
            if action == "DIG" or _units(tape[hour])[worker][0] in MOVES or _units(tape[hour])[worker][0] == "PASS":
                _replace_unit(tape[hour], worker, [action])
        routes_changed += 1
    local_plan, blocked = plan_idle_digs(farm, tape)
    return local_plan, blocked, routes_changed


def delay_blocked_weeds(farm: dict, tape: list[dict]) -> tuple[dict, list[dict], int]:
    """Insert a DIG before blocked work and absorb a one-hour slip in PASS.

    Preserve the worker's action order and the next position checkpoint. Never
    shift HARVEST/COLLECT_FERTILIZER: their exact product-output hour is part
    of the replay contract. Other workers and market orders are untouched.
    """
    changed = 0
    attempted: set[tuple[int, int]] = set()
    for _ in range(20):
        local_plan, blocked = plan_idle_digs(farm, tape)
        target = next((row for row in blocked if tuple(row["xy"]) not in attempted), None)
        if target is None:
            return local_plan, blocked, changed
        xy = tuple(target["xy"])
        worker = int(target["worker"])
        deadline = int(target["deadline_hour"])
        if _delay_one_worker(tape, worker, deadline, xy):
            changed += 1
            attempted.clear()
        else:
            attempted.add(xy)
    local_plan, blocked = plan_idle_digs(farm, tape)
    return local_plan, blocked, changed


def _delay_one_worker(tape: list[dict], worker: int, deadline: int,
                      xy: tuple[int, int]) -> bool:
    if deadline >= len(tape) or worker >= len(tape[deadline].get("positions") or []):
        return False
    first = _units(tape[deadline])[worker]
    if not first or first[0] not in EMPTY_REQUIRED or tuple(tape[deadline]["positions"][worker]) != xy:
        return False
    return _insert_worker_delay(tape, worker, deadline, xy, ["DIG"])


def _insert_worker_delay(tape: list[dict], worker: int, deadline: int,
                         xy: tuple[int, int], inserted: list,
                         protected_slots: set[tuple[int, int]] | None = None,
                         detail: dict | None = None, *,
                         allow_output_shift: bool = False,
                         allow_terminal_move: bool = False) -> bool:
    """Insert one action while preserving the worker's remaining daily jobs.

    The ordinary mode is deliberately strict: the delay must disappear into a
    later PASS before any yield or bag-transfer action moves.  The deadline
    worker may use ``allow_output_shift`` as a last source-worker fallback.  In
    that mode all original jobs retain their order and remain in the same day;
    only their intraday hour changes.  A final pure movement may also be
    discarded because farm hands are dismissed after hour 23.
    """
    protected_slots = protected_slots or set()
    for end in range(deadline + 1, len(tape)):
        if any(worker >= len(tape[h].get("positions") or []) for h in range(deadline, end + 1)):
            break
        if any((h, worker) in protected_slots for h in range(deadline, end + 1)):
            break
        actions = [_units(tape[h])[worker] for h in range(deadline, end + 1)]
        if any(not action for action in actions):
            break
        if (not allow_output_shift
                and any(action[0] in ("HARVEST", "COLLECT_FERTILIZER", "PICKUP", "DROP")
                        for action in actions[:-1])):
            break
        absorbs_pass = actions[-1] == ["PASS"]
        absorbs_terminal_move = (
            allow_terminal_move and end == len(tape) - 1 and actions[-1][0] in MOVES
        )
        if not absorbs_pass and not absorbs_terminal_move:
            continue
        positions = []
        pos = xy
        for action in [inserted, *actions[:-1]]:
            positions.append(pos)
            dx, dy = MOVES.get(action[0], (0, 0))
            pos = (pos[0] + dx, pos[1] + dy)
        # A fulfilled HIRE sees the post-field positions on this hour.  Do not
        # let a shifted farmer/hand alter any later source worker's spawn tile.
        spawn_safe = True
        for offset, hour in enumerate(range(deadline, end + 1)):
            if hour + 1 >= len(tape):
                continue
            now_positions = tape[hour].get("positions") or []
            next_positions = tape[hour + 1].get("positions") or []
            if len(next_positions) <= len(now_positions) or worker >= len(next_positions):
                continue
            action = inserted if offset == 0 else actions[offset - 1]
            px, py = positions[offset]
            dx, dy = MOVES.get(action[0], (0, 0))
            if (px + dx, py + dy) != tuple(next_positions[worker]):
                spawn_safe = False
                break
        if not spawn_safe:
            continue
        if end + 1 < len(tape) and worker < len(tape[end + 1].get("positions") or []):
            if pos != tuple(tape[end + 1]["positions"][worker]):
                continue
        original_events = []
        for hour in range(deadline, end + 1):
            events = tape[hour].get("unit_events") or []
            original_events.append(dict(events[worker]) if worker < len(events) else {"seeds": {}, "shed": {}})
        for offset, hour in enumerate(range(deadline, end + 1)):
            step = tape[hour]
            step["positions"][worker] = list(positions[offset])
            _replace_unit(step, worker, list(inserted) if offset == 0 else list(actions[offset - 1]))
            events = step.get("unit_events")
            if events is not None and worker < len(events):
                events[worker] = _inserted_event(inserted) if offset == 0 else original_events[offset - 1]
                for bucket, field in (("seeds", "seed_delta_units"), ("shed", "shed_delta_units")):
                    totals = {}
                    for event in events:
                        for item, delta in (event.get(bucket) or {}).items():
                            totals[item] = int(totals.get(item) or 0) + int(delta)
                    step[field] = totals
        if detail is not None:
            detail.update({
                "absorbed_hour": end,
                "absorbed_action": list(actions[-1]),
                "absorption": "pass" if absorbs_pass else "terminal_move",
                "shifted_hours": end - deadline,
            })
        return True
    return False


def repair_first_arrival_weeds(farm: dict, tape: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Give every source worker an earliest-arrival chance to clear a weed.

    A PASS at the tile becomes a conditional live DIG.  Otherwise the current
    action is shifted one hour at a time until a later PASS absorbs the added
    DIG.  Visits are tried in ``(hour, worker index)`` order, which is also the
    official engine's unit execution order.  If every conservative attempt
    fails, the original PLANT/BUILD worker gets one final whole-day attempt:
    DIG first, then keep all remaining source jobs in order, absorbing the
    delay in a later PASS or an irrelevant terminal movement.  A supplementary
    worker is considered by the caller only after that final attempt fails.
    """
    plan: dict[tuple[int, int], tuple[int, int]] = {}
    repairs: list[dict] = []
    blocked: list[dict] = []
    processed: set[tuple[int, int]] = set()

    # A successful insertion can shift a later operation by one hour, so
    # recompute outstanding deadlines after every target instead of relying on
    # a stale dawn snapshot.
    for _ in range(100):
        needs = _weed_needs(farm, tape)
        pending = sorted(
            ((deadline, xy) for xy, deadline in needs.items() if xy not in processed),
            key=lambda row: (row[0], row[1]),
        )
        if not pending:
            break
        deadline, xy = pending[0]
        processed.add(xy)
        if _has_source_dig(tape, xy, deadline):
            repairs.append({"xy": list(xy), "deadline_hour": deadline[0],
                            "deadline_worker": deadline[1], "mode": "source_dig"})
            continue

        visits = []
        for hour, step in enumerate(tape):
            positions = step.get("positions") or []
            for worker, action in enumerate(_units(step)):
                when = (hour, worker)
                if when > deadline or worker >= len(positions) or tuple(positions[worker]) != xy:
                    continue
                visits.append((when, list(action) if isinstance(action, list) and action else ["PASS"]))

        tried = []
        resolved = False
        for (hour, worker), action in visits:
            tried.append({"hour": hour, "worker": worker, "action": action[0] if action else "PASS"})
            if action == ["DIG"] and (hour, worker) < deadline:
                repairs.append({"xy": list(xy), "deadline_hour": deadline[0],
                                "deadline_worker": deadline[1], "visit_hour": hour,
                                "worker": worker, "mode": "source_dig"})
                resolved = True
                break
            if action == ["PASS"] and (hour, worker) < deadline and (hour, worker) not in plan:
                plan[(hour, worker)] = xy
                repairs.append({"xy": list(xy), "deadline_hour": deadline[0],
                                "deadline_worker": deadline[1], "visit_hour": hour,
                                "worker": worker, "mode": "idle_pass", "absorbed_hour": hour})
                resolved = True
                break
            detail = {}
            if _insert_worker_delay(
                tape, worker, hour, xy, ["DIG"], protected_slots=set(plan), detail=detail,
            ):
                repairs.append({"xy": list(xy), "deadline_hour": deadline[0],
                                "deadline_worker": deadline[1], "visit_hour": hour,
                                "worker": worker, "mode": "shift_into_pass", **detail})
                resolved = True
                break
        if not resolved:
            # The original planter/builder is not merely another failed scan
            # candidate.  It can DIG on its own deadline and move its complete
            # remaining schedule one hour later, provided no productive job is
            # lost from the day and source-worker spawn checkpoints stay fixed.
            detail = {}
            deadline_hour, deadline_worker = deadline
            if _insert_worker_delay(
                tape, deadline_worker, deadline_hour, xy, ["DIG"],
                protected_slots=set(plan), detail=detail,
                allow_output_shift=True, allow_terminal_move=True,
            ):
                repairs.append({
                    "xy": list(xy), "deadline_hour": deadline_hour,
                    "deadline_worker": deadline_worker,
                    "visit_hour": deadline_hour, "worker": deadline_worker,
                    "mode": "deadline_worker_full_day_shift", **detail,
                })
                resolved = True
        if not resolved:
            job = list(_units(tape[deadline[0]])[deadline[1]])
            blocked.append({"xy": list(xy), "deadline_hour": deadline[0],
                            "worker": deadline[1], "job": job,
                            "reason": "no_arrival_can_absorb_dig",
                            "arrivals_tried": tried})
    return plan, blocked, repairs


def _inserted_event(action: list) -> dict:
    seeds = {str(action[1]): -1} if len(action) >= 2 and action[0] == "PLANT" else {}
    return {"seeds": seeds, "shed": {}}


def _recompute_step_events(step: dict) -> None:
    events = step.get("unit_events")
    if events is None:
        return
    for bucket, field in (("seeds", "seed_delta_units"), ("shed", "shed_delta_units")):
        totals = {}
        for event in events:
            for item, delta in (event.get(bucket) or {}).items():
                totals[item] = int(totals.get(item) or 0) + int(delta)
        step[field] = totals


def _replace_idle_action(step: dict, worker: int, action: list) -> None:
    _replace_unit(step, worker, list(action))
    events = step.get("unit_events")
    if events is not None and worker < len(events):
        events[worker] = _inserted_event(action)
        _recompute_step_events(step)


def _try_worker_sequence(tape: list[dict], worker: int, start: int,
                         xy: tuple[int, int], inserted: list[list]) -> tuple[bool, list[dict]]:
    """Transactionally insert several stationary recovery actions."""
    trial = copy.deepcopy(tape)
    details = []
    for offset, action in enumerate(inserted):
        hour = start + offset
        if (hour >= len(trial) or worker >= len(trial[hour].get("positions") or [])
                or tuple(trial[hour]["positions"][worker]) != xy):
            return False, []
        original = _units(trial[hour])[worker]
        if original == ["PASS"]:
            _replace_idle_action(trial[hour], worker, action)
            details.append({"hour": hour, "action": list(action), "mode": "replace_pass",
                            "absorbed_hour": hour})
            continue
        detail = {}
        if not _insert_worker_delay(trial, worker, hour, xy, action, detail=detail):
            return False, []
        details.append({"hour": hour, "action": list(action), "mode": "shift_into_pass", **detail})
    tape[:] = trial
    return True, details


def _job_present(tile, job: list) -> bool:
    if not isinstance(tile, dict) or not job:
        return False
    if job[0] == "PLANT":
        return tile.get("kind") == "PLANT" and len(job) >= 2 and tile.get("crop") == job[1]
    if job[0] == "BUILD_COOP":
        return tile.get("kind") == "COOP"
    if job[0] == "BUILD_PASTURE":
        return tile.get("kind") == "PASTURE"
    return False


def repair_deferred_jobs(farm: dict, tape: list[dict], jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """On the mandatory recovery day, scan every passer before hiring.

    A recovery sequence removes the weed, performs the originally blocked
    PLANT/BUILD, and waters a newly planted crop so it survives midnight.
    Existing workers are considered in first-arrival order.  The operation is
    transactional: partial insertions are never left on the tape.
    """
    repaired, remaining = [], []
    tiles = farm.get("tiles") or []
    for job_row in jobs:
        row = copy.deepcopy(job_row)
        xy = tuple(int(v) for v in row["xy"])
        job = list(row.get("job") or [])
        if not job or job[0] not in EMPTY_REQUIRED:
            row["reason"] = "unsupported_deferred_job"
            remaining.append(row)
            continue
        x, y = xy
        tile = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else "LOCKED"
        if _job_present(tile, job):
            repaired.append({**row, "mode": "already_present"})
            continue
        if tile not in (None, "WEED") and not is_weed(tile):
            row["reason"] = "deferred_tile_occupied"
            remaining.append(row)
            continue
        sequence = (["DIG"],) if is_weed(tile) else ()
        actions = [list(a) for a in sequence] + [job]
        if job[0] == "PLANT":
            actions.append(["WATER"])
        visits = []
        for hour, step in enumerate(tape):
            for worker, pos in enumerate(step.get("positions") or []):
                if tuple(pos) == xy:
                    visits.append((hour, worker))
        success = False
        for hour, worker in visits:
            ok, details = _try_worker_sequence(tape, worker, hour, xy, actions)
            if ok:
                repaired.append({**row, "mode": "existing_worker_sequence",
                                 "visit_hour": hour, "worker": worker, "actions": details})
                success = True
                break
        if not success:
            row["reason"] = "no_passer_can_absorb_recovery_sequence"
            row["arrivals_tried"] = [{"hour": h, "worker": w} for h, w in visits]
            remaining.append(row)
    return repaired, remaining


def _fulfilled_source_hire_hours(tape: list[dict]) -> tuple[list[int], int]:
    """Actual source births, including idle hands that affect later spawns."""
    source_hands = max(
        (max(0, len(step.get("positions") or []) - 1) for step in (tape or [])),
        default=0,
    )
    hours = []
    for hand_index in range(source_hands):
        birth = next(
            (hour for hour, step in enumerate(tape or [])
             if len(step.get("positions") or []) > hand_index + 1),
            None,
        )
        if birth is not None:
            hours.append(max(0, int(birth) - 1))
    return hours, source_hands


def _recovery_dependency_hour(tape: list[dict], xy: tuple[int, int], job: list) -> int | None:
    """First source operation that requires the recovered tile to exist.

    A recovered crop receives its own WATER, so an earlier source WATER can be
    safely missed. Fertilizing, harvesting, or digging that crop cannot. A
    recovered building likewise has to exist before the source starts using or
    removing it.
    """
    if not job:
        return None
    if job[0] == "PLANT":
        dependent = {"FERTILIZE", "HARVEST", "DIG"}
    elif job[0] in ("BUILD_COOP", "BUILD_PASTURE"):
        dependent = {"PLACE", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER", "DIG"}
    else:
        return None
    for hour, step in enumerate(tape):
        positions = step.get("positions") or []
        for worker, action in enumerate(_units(step)):
            if (worker < len(positions) and tuple(positions[worker]) == xy
                    and action and action[0] in dependent):
                return hour
    return None


def plan_deferred_extra_hand(farm: dict, tape: list[dict], jobs: list[dict], board_size: int = 10) -> dict | None:
    """Hire one mandatory recovery hand after all source hires.

    With no same-day deadline left, the appended worker can safely preserve all
    source indices, walk to each deferred tile, DIG, perform the blocked job,
    and WATER a planted crop before midnight.
    """
    if not tape or not jobs:
        return None
    source_hire_hours, source_hands = _fulfilled_source_hire_hours(tape)
    latest_source_hire = max(source_hire_hours, default=0)
    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    candidate = None
    for hire_hour in range(latest_source_hire, len(tape) - 1):
        if len(tape[hire_hour].get("market") or []) >= 10:
            continue
        positions = tape[hire_hour + 1].get("positions") or []
        if not positions or len(positions) - 1 < source_hands:
            continue
        proxy = {"farmer": positions[0], "hands": positions[1:]}
        start = tuple(engine._spawn_hand(proxy, board_size))
        x, y = start
        hour = hire_hour + 1
        actions, dig_targets, dig_hours = {}, {}, []
        feasible = True
        for row in jobs:
            xy = tuple(int(v) for v in row["xy"])
            job = list(row.get("job") or [])
            if not job or job[0] not in EMPTY_REQUIRED:
                feasible = False
                break
            while (x, y) != xy and hour < len(tape):
                if x < xy[0]:
                    action, x = "EAST", x + 1
                elif x > xy[0]:
                    action, x = "WEST", x - 1
                elif y < xy[1]:
                    action, y = "SOUTH", y + 1
                else:
                    action, y = "NORTH", y - 1
                actions[hour] = [action]
                hour += 1
            if (x, y) != xy or hour >= len(tape):
                feasible = False
                break
            tile = farm["tiles"][xy[1]][xy[0]]
            if is_weed(tile):
                actions[hour] = ["DIG"]
                dig_targets[hour] = list(xy)
                dig_hours.append(hour)
                hour += 1
            if hour >= len(tape):
                feasible = False
                break
            dependency_hour = _recovery_dependency_hour(tape, xy, job)
            # This appended hand acts after every source worker. Completing the
            # job in the same hour as FERTILIZE/PLACE/etc. is already too late.
            if dependency_hour is not None and hour >= dependency_hour:
                feasible = False
                break
            actions[hour] = job
            hour += 1
            if job[0] == "PLANT":
                if hour >= len(tape):
                    feasible = False
                    break
                actions[hour] = ["WATER"]
                hour += 1
        if feasible:
            targets = [list(map(int, row["xy"])) for row in jobs]
            candidate = {"hire_hour": hire_hour, "spawn": list(start), "xy": targets[0],
                         "targets": targets, "dig_hour": dig_hours[0] if dig_hours else None,
                         "dig_hours": dig_hours, "dig_targets": dig_targets,
                         "handoffs": {}, "actions": actions, "mandatory_recovery": True,
                         "source_hands_before_extra": source_hands,
                         "recovery_jobs": copy.deepcopy(jobs)}
    # Hire as late as the complete recovery sequence safely permits. This
    # avoids paying early and leaves earlier market slots to source staffing
    # and truly immediate inputs.
    return candidate


def repair_same_hour_build_place(tape: list[dict]) -> list[dict]:
    """Let the animal carrier wait for a weed-delayed builder.

    Swapping BUILD/PLACE between workers would lose the animal in the original
    worker's bag. Instead retain its owner and delay that owner's non-yield
    segment into one later PASS. Harvest and warehouse-transfer hours cannot
    be moved by this repair.
    """
    repaired = []
    for _ in range(20):
        changed = False
        for hour, step in enumerate(tape):
            positions = step.get("positions") or []
            units = _units(step)
            for worker, action in enumerate(units):
                if (len(action) < 2 or action[0] != "PLACE" or worker >= len(positions)
                        or action[1] not in ("COW", "SHEEP", "GOOSE")):
                    continue
                xy = tuple(positions[worker])
                build = "BUILD_COOP" if action[1] == "GOOSE" else "BUILD_PASTURE"
                builder = next((i for i in range(worker + 1, len(units))
                                if i < len(positions) and tuple(positions[i]) == xy
                                and units[i] == [build]), None)
                if builder is not None and _insert_worker_delay(tape, worker, hour, xy, ["PASS"]):
                    repaired.append({"hour": hour, "worker": worker, "builder": builder,
                                     "xy": list(xy), "animal": action[1]})
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return repaired


def plan_extra_hand(tape: list[dict], xy: tuple[int, int], deadline: tuple[int, int], board_size: int = 10) -> dict | None:
    """One extra hand, hired after all source hires, to clear one blocked tile.

    This preserves every source hand's index. It is only a schedule candidate:
    the caller must still check cash, the actual hire fill, and the official
    target after replay. HIRE takes effect in the following hour.
    """
    return plan_extra_hand_targets(tape, [{"xy": list(xy), "deadline_hour": deadline[0],
                                           "worker": deadline[1]}], board_size)


def plan_extra_hand_targets(tape: list[dict], targets: list[dict], board_size: int = 10) -> dict | None:
    """Route one appended hand through all dawn-visible blocking weeds.

    The hand is hired only after source hires, and each DIG must finish before
    the corresponding source worker's first PLANT/BUILD. If it arrives exactly
    at that hour, the source worker can DIG and this later-index hand can do
    the original PLANT/BUILD in the same turn, provided no intervening source
    job depends on it. No production hour or later worker route is moved.
    """
    if not tape or not targets:
        return None
    deadlines: dict[tuple[int, int], int] = {}
    target_workers = {}
    for target in targets:
        xy = tuple(int(v) for v in target["xy"])
        hour = int(target["deadline_hour"])
        if xy not in deadlines or hour < deadlines[xy]:
            deadlines[xy] = hour
            target_workers[xy] = int(target["worker"])
    ordered = sorted(deadlines.items(), key=lambda row: (row[1], row[0]))
    source_hire_hours, source_hands = _fulfilled_source_hire_hours(tape)
    latest_source_hire = max(source_hire_hours, default=0)
    candidate = None
    for hire_hour in range(latest_source_hire, min(ordered[0][1], len(tape) - 1)):
        if len(tape[hire_hour].get("market") or []) >= 10:
            continue
        positions = tape[hire_hour + 1].get("positions") or []
        if not positions or len(positions) - 1 < source_hands:
            continue
        # The official spawn is the least-occupied shed access tile after the
        # source hires at this hour, before the extra hire is appended.
        from kaggle_environments.envs.kaggriculture import kaggriculture as engine

        proxy = {"farmer": positions[0], "hands": positions[1:]}
        start = tuple(engine._spawn_hand(proxy, board_size))
        actions = {}
        handoffs = {}
        x, y = start
        hour = hire_hour + 1
        dig_hours = []
        feasible = True
        for xy, deadline_hour in ordered:
            while (x, y) != xy and hour < deadline_hour:
                if x < xy[0]:
                    action, x = "EAST", x + 1
                elif x > xy[0]:
                    action, x = "WEST", x - 1
                elif y < xy[1]:
                    action, y = "SOUTH", y + 1
                else:
                    action, y = "NORTH", y - 1
                actions[hour] = [action]
                hour += 1
            if (x, y) != xy or hour > deadline_hour:
                feasible = False
                break
            if hour == deadline_hour:
                step = tape[hour]
                worker = target_workers[xy]
                units = _units(step)
                positions = step.get("positions") or []
                if (worker >= len(units) or worker >= len(positions)
                        or tuple(positions[worker]) != xy or not units[worker]
                        or units[worker][0] not in EMPTY_REQUIRED
                        or any(i < len(positions) and tuple(positions[i]) == xy
                               and action and action[0] not in ("PASS", *MOVES)
                               for i, action in enumerate(units) if i > worker)):
                    feasible = False
                    break
                actions[hour] = list(units[worker])
                handoffs[hour] = {"worker": worker, "xy": list(xy), "job": list(units[worker])}
            else:
                actions[hour] = ["DIG"]
            dig_hours.append(hour)
            hour += 1
        if feasible:
            candidate = {"hire_hour": hire_hour, "dig_hour": dig_hours[0],
                         "spawn": list(start), "xy": list(ordered[0][0]),
                         "targets": [list(xy) for xy, _ in ordered],
                         "dig_hours": dig_hours,
                         "dig_targets": {h: list(xy) for h, (xy, _) in zip(dig_hours, ordered)},
                         "handoffs": handoffs,
                         "actions": actions,
                         "source_hands_before_extra": source_hands}
    # The latest feasible hire minimizes cash lock-up and avoids crowding the
    # source's dawn staffing block, while still clearing before (or handing off
    # exactly at) every original PLANT/BUILD deadline.
    return candidate
