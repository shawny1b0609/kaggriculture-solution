"""Opt-in execution v2. Frozen v1 screening and price timing stay untouched.

Only own live state and precompiled worker instructions are inspected. Delays
must fit the day and preserve later worker births. Node matching is unchanged.
"""
from __future__ import annotations

import copy

from tools.node_contract_agent import ContractRepairMixin
from runtime.grid_agent import own_farm, engine, _estimate_market_cost
from runtime.resource_contract import dawn_requirements, contract_order_gaps
from runtime.weed_repair import (_insert_worker_delay, _units, _replace_unit,
                                 _recompute_step_events, MOVES)

ANIMALS = {'SHEEP', 'COW', 'GOOSE'}
FIELD = {'DIG', 'PLANT', 'BUILD_COOP', 'BUILD_PASTURE', 'HARVEST',
         'PLACE', 'FEED', 'CARE', 'WATER', 'FERTILIZE', 'COLLECT_FERTILIZER'}


def delay_worker_chain(tape, worker, hour, inserted, protected=(), *, dependency=None):
    """Transactional one-hour delay with cross-worker dependency protection.

    Bag ownership is preserved. Reject a shift if another worker operates on
    the affected tile between an original job and its new execution time.
    The one explicitly repaired BUILD -> PLACE inversion is the exception.
    """
    trial = copy.deepcopy(tape)
    positions = tape[hour].get('positions') or []
    if worker >= len(positions):
        return None
    detail = {}
    if not _insert_worker_delay(trial, worker, hour, tuple(positions[worker]), inserted,
            protected_slots=set(protected), detail=detail, allow_output_shift=True,
            allow_terminal_move=worker > 0):
        return None
    end = detail['absorbed_hour']
    for h in range(hour, end):
        action = _units(tape[h])[worker]
        if action[0] not in FIELD:
            continue
        xy = tuple(tape[h]['positions'][worker])
        for t in (h, h + 1):
            for other, op in enumerate(_units(tape[t])):
                if other == worker or other >= len(tape[t].get('positions') or []):
                    continue
                if (h, worker) < (t, other) < (h + 1, worker):
                    if op[0] in FIELD and tuple(tape[t]['positions'][other]) == xy:
                        if dependency != (h, worker, t, other):
                            return None
    tape[:] = trial
    return detail


def repair_build_carriers(tape, protected=()):
    repaired, blocked = [], []
    for _ in range(24):
        changed = False
        for hour, step in enumerate(tape):
            units, positions = _units(step), step.get('positions') or []
            for worker, action in enumerate(units):
                if len(action) < 2 or action[0] != 'PLACE' or action[1] not in ANIMALS:
                    continue
                if worker >= len(positions):
                    continue
                build = 'BUILD_COOP' if action[1] == 'GOOSE' else 'BUILD_PASTURE'
                builder = next((w for w in range(worker + 1, len(units))
                    if w < len(positions) and positions[w] == positions[worker]
                    and units[w] == [build]), None)
                if builder is None:
                    continue
                row = {'hour': hour, 'worker': worker, 'builder': builder,
                       'xy': positions[worker], 'animal': action[1]}
                detail = delay_worker_chain(tape, worker, hour, ['PASS'], protected,
                                          dependency=(hour, worker, hour, builder))
                if detail:
                    repaired.append({**row, **detail})
                    changed = True
                    break
                blocked.append(row)
            if changed:
                break
        if not changed:
            break
    return repaired, blocked


def wait_for_resources(obs, tape, hour, protected=()):
    """Do not walk away empty when a productive pickup can still fit today."""
    if hour >= len(tape):
        return [], []
    farm, private = copy.deepcopy(own_farm(obs)), copy.deepcopy(obs.get('private') or {})
    positions = [farm.get('farmer') or [], *farm.get('hands', [])]
    repaired, blocked = [], []
    for worker, initial in enumerate(_units(tape[hour])):
        expected = tape[hour].get('positions') or []
        if worker >= len(positions) or worker >= len(expected):
            continue
        action = initial
        if (action[0] == 'PICKUP' and len(action) >= 3
                and positions[worker] == expected[worker]
                and engine._is_shed_adjacent(tuple(positions[worker]), len(farm['tiles']))):
            held = int((private.get('shed') or {}).get(action[1]) or 0)
            # A partial pickup must not be repeated at its full original size.
            # Wait instead, retaining the whole contract for the next market.
            if held < int(action[2]):
                row = {'worker': worker, 'hour': hour, 'item': action[1],
                       'need': int(action[2]), 'have': held}
                detail = delay_worker_chain(tape, worker, hour, ['PASS'], protected)
                if detail:
                    repaired.append({**row, **detail})
                    action = ['PASS']
                else:
                    blocked.append(row)
        engine._apply_unit_action(farm, private, worker, action,
                                 len(farm['tiles']), int(obs['day']), 24, 100)
    return repaired, blocked


def locked_market_capacity(tape, hour, projected, hard_buys):
    """Room to release now if next hour has no sale slot.

    Source own-worker stock deltas and staffing are known instructions, not
    future market information. Never assume that an unsubmitted sale fills.
    """
    if hour + 2 >= len(tape):
        return 0
    held = copy.deepcopy(projected)
    for op, item, qty in hard_buys:
        bucket = held.setdefault('seeds' if op == 'BUY_SEED' else 'shed', {})
        bucket[item] = int(bucket.get(item) or 0) + qty
    step = tape[hour + 1]
    for event in step.get('unit_events') or []:
        for name in ('shed', 'seeds'):
            for item, delta in (event.get(name) or {}).items():
                held.setdefault(name, {})[item] = max(0, int(held.get(name, {}).get(item) or 0) + int(delta))
    orders, _ = contract_order_gaps(dawn_requirements(tape[hour + 2:hour + 3]), held)
    hires = max(0, int(step.get('post_hands') or 0) - max(0, len(step.get('positions') or []) - 1))
    lands = max(0, int(step.get('post_land') or 0) - int(step.get('land') or 1))
    if hires + lands + len(orders) < 10:
        return 0
    purchases = sum(q for op, _, q in orders if op != 'BUY_SEED')
    overflow = max(0, sum(held.get('shed', {}).values()) + purchases - 100)
    return overflow


class ResilientContractMixin(ContractRepairMixin):
    """Use before GridAgent / RandomReplayAgent in the class MRO."""

    def _dawn(self, obs, day):
        super()._dawn(obs, day)
        if self.tape:
            repaired, blocked = repair_build_carriers(self.tape, self.weed_plan)
            for row in repaired:
                self._choice_log('execution_v2', {'day': day, 'kind': 'build_carrier_chain', **row})
            for row in blocked:
                self._choice_log('execution_v2_unresolved', {'day': day, 'kind': 'build_carrier_chain', **row})

    def _play(self, obs, hour):
        repaired, blocked = wait_for_resources(obs, self.tape, hour, self.weed_plan)
        for row in repaired:
            self._choice_log('execution_v2', {'day': obs['day'], 'kind': 'resource_chain_wait', **row})
        for row in blocked:
            self._choice_log('execution_v2_unresolved', {'day': obs['day'], 'kind': 'resource_chain_wait', **row})
        return super()._play(obs, hour)

    def _schedule_current_first_market(self, obs, projected, hard_buys, staffing,
                                       future_buys, stale_buys, sales, future_staffing,
                                       *, turn, day, hour, protected_cash_after_hard=None):
        overflow = locked_market_capacity(self.tape, hour, projected, hard_buys)
        if overflow and self.sell_mode == 'seller':
            reserved = dawn_requirements(self.tape[hour + 1:])['shed']
            # Preserve existing signals; only add a real capacity obligation.
            free_room = max(0, 100 - sum(projected.get('shed', {}).values()))
            cost = _estimate_market_cost(obs, hard_buys + staffing)
            plan = self.market_seller.plan(obs, projected.get('shed') or {}, reserved,
                minimum_cash=cost + float(protected_cash_after_hard or 0),
                capital_floor=self.market_capital_floor,
                required_capacity=free_room + overflow,
                available_order_slots=max(0, 10 - len(hard_buys) - len(staffing)))
            proposed = {o[1]: list(o) for o in sales}
            for order in plan.orders:
                if order[1] not in proposed or order[2] > proposed[order[1]][2]:
                    proposed[order[1]] = list(order)
            # Capacity-releasing signals first, still only spare farm slots.
            order_keys = [o[1] for o in plan.orders] + [o[1] for o in sales if o[1] not in {x[1] for x in plan.orders}]
            sales[:] = [proposed[key] for key in order_keys]
            self._choice_log('execution_v2', {'day': day, 'hour': hour,
                'kind': 'release_before_locked_market', 'required_release': overflow})
        return super()._schedule_current_first_market(obs, projected, hard_buys, staffing,
            future_buys, stale_buys, sales, future_staffing, turn=turn, day=day, hour=hour,
            protected_cash_after_hard=protected_cash_after_hard)
