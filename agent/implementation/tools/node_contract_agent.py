"""Opt-in node-contract repairs, isolated from the frozen screening runtime.

Only known source actions and live stock/prices are used. No future shop,
opponent or RNG information is consulted. No node matching is relaxed.
"""
from __future__ import annotations

import copy

from runtime.grid_agent import (_estimate_market_cost, own_farm, engine,
                                plan_source_hire_schedule,
                                remove_dead_animal_pickups, trim_consumable_pickups)
from runtime.resource_contract import dawn_requirements, contract_order_gaps
from runtime.weed_repair import _insert_worker_delay, MOVES


def postpone_missing_pickups(obs, tape, hour, protected_slots=()):
    """Wait at the shed, preserving the worker's jobs and all birth positions.

    Only maintenance-only suffixes may move; a productive/structural action
    is a hard barrier. The original PICKUP remains the next hour's
    demand, so the existing buyer retries it and the seller reserves its stock.
    """
    if hour >= len(tape):
        return []
    farm = own_farm(obs)
    live_positions = [farm.get('farmer') or [], *(farm.get('hands') or [])]
    private = copy.deepcopy(obs.get('private') or {})
    simulated = copy.deepcopy(farm)
    safe_ops = set(MOVES) | {'PASS', 'PICKUP', 'FEED', 'CARE', 'WATER', 'FERTILIZE'}
    repairs = []
    actions = [tape[hour].get('farmer') or ['PASS'], *(tape[hour].get('hands') or [])]
    for worker, action in enumerate(actions):
        positions = tape[hour].get('positions') or []
        if (action[:1] == ['PICKUP'] and len(action) >= 3
                and worker < len(live_positions) and worker < len(positions)
                and list(live_positions[worker]) == list(positions[worker])
                and engine._is_shed_adjacent(tuple(live_positions[worker]), len(farm['tiles']))
                and int((private.get('shed') or {}).get(action[1]) or 0) < int(action[2])):
            protected = set(protected_slots)
            for later in range(hour + 1, len(tape)):
                units = [tape[later].get('farmer') or ['PASS'], *(tape[later].get('hands') or [])]
                if worker < len(units) and units[worker][0] not in safe_ops:
                    protected.add((later, worker))
            trial = copy.deepcopy(tape)
            detail = {}
            if _insert_worker_delay(trial, worker, hour, tuple(positions[worker]), ['PASS'],
                    protected_slots=protected, detail=detail, allow_output_shift=True,
                    allow_terminal_move=worker > 0):
                tape[:] = trial
                repairs.append({'worker': worker, 'hour': hour, 'item': action[1],
                                'quantity': action[2], **detail})
                action = ['PASS']
        # Earlier workers can consume or deposit shed stock in this field phase.
        if worker < len(live_positions):
            engine._apply_unit_action(simulated, private, worker, action,
                                     len(farm['tiles']), int(obs.get('day') or 0), 24, 100)
    return repairs


class ContractRepairMixin:
    def _play(self, obs, hour):
        repairs = postpone_missing_pickups(obs, self.tape, hour, set(self.weed_plan))
        for repair in repairs:
            self._choice_log('contract_repair', {'turn': int(obs['day']) * 24 + hour,
                            'kind': 'wait_for_missing_pickup', **repair})
        return super()._play(obs, hour)

    def _schedule_current_first_market(self, obs, projected, hard_buys, staffing,
                                       future_buys, stale_buys, sales, future_staffing,
                                       *, turn, day, hour, protected_cash_after_hard=None):
        # Tomorrow's full farm-order block can prohibit liquidation even when
        # the shed contains valuable products deposited at midnight. Advance
        # an input order today, NEVER displace tomorrow's original worker births.
        if (getattr(self, 'contract_bridge_enabled', True) and hour == 23
                and day < 29 and self.preselected_edge is not None):
            tape = self.tapes.day_tape(str(self.preselected_edge['example']), day + 1)
            remove_dead_animal_pickups(tape)
            trim_consumable_pickups(tape)
            board_size = len(own_farm(obs).get('tiles') or []) or 10
            hires = sum(h == 0 for h in plan_source_hire_schedule(tape, board_size)['hours'])
            lands = max(0, int(tape[0].get('post_land') or 0)
                        - len(own_farm(obs).get('unlocked_quadrants') or [])
                        - sum(o[0] == 'BUY_LAND' for o in staffing)) if tape else 0
            held = copy.deepcopy(projected)
            for op, item, qty in hard_buys:
                bucket = held.setdefault('seeds' if op == 'BUY_SEED' else 'shed', {})
                bucket[item] = int(bucket.get(item) or 0) + qty
            engine._drop_inventories_to_shed(held, 100)
            contract = dawn_requirements(tape[:2], rely_on_intraday_buys=False)
            gaps, _ = contract_order_gaps(contract, held)
            # A full order book alone is not a liquidity emergency. Leave a
            # solvent source schedule untouched (including late-game inputs).
            next_obs = copy.deepcopy(obs)
            next_farm = own_farm(next_obs)
            next_farm['hands'] = []
            next_cost = _estimate_market_cost(next_obs, [['HIRE']]*hires + [['BUY_LAND']]*lands + gaps)
            next_cash = (float(own_farm(obs).get('money') or 0)
                         + self.market_seller.stressed_revenue(obs, sales)
                         - _estimate_market_cost(obs, hard_buys + staffing))
            if gaps and hires + lands + len(gaps) >= 10 and next_cash < next_cost:
                merged = {(o[0], o[1]): list(o) for o in hard_buys}
                for op, item, qty in gaps:
                    if (op, item) in merged:
                        merged[op, item][2] += qty
                    else:
                        merged[op, item] = [op, item, qty]
                candidate = list(merged.values())
                reserved = dict(contract.get('shed') or {})
                slots = max(0, 10 - len(candidate) - len(staffing))
                cost = _estimate_market_cost(obs, candidate + staffing)
                plan = self.market_seller.plan(obs, projected.get('shed') or {}, reserved,
                    minimum_cash=cost, available_order_slots=slots,
                    required_capacity=sum(o[2] for o in candidate if o[0] != 'BUY_SEED'))
                sale_orders = plan.orders
                room = 100 - sum((projected.get('shed') or {}).values())
                room += sum(o[2] for o in sale_orders)
                needed = sum(o[2] for o in candidate if o[0] != 'BUY_SEED')
                midnight_drop = sum(max(0, int(qty or 0))
                                    for bag in projected.get('inventories') or []
                                    for qty in (bag or {}).values())
                affordable = cost <= float(own_farm(obs).get('money') or 0) + self.market_seller.stressed_revenue(obs, sale_orders)
                if affordable and needed + midnight_drop <= room and len(candidate) + len(staffing) <= 10:
                    hard_buys = candidate
                    # Preserve the caller's shared list so planned/issued sale
                    # accounting reflects the revised obligation as well.
                    sales[:] = sale_orders
                    keys = set(merged)
                    future_buys = [o for o in future_buys if (o[0], o[1]) not in keys]
                    stale_buys = [o for o in stale_buys if (o[0], o[1]) not in keys]
                    for op, item, qty in gaps:
                        total = merged[op, item][2]
                        self._queue_market_order(op, item, total, due_turn=turn,
                            priority=0, reason='next_dawn_liquidity_slot',
                            planned_turn=turn, hard_quantity=total)
                    self._choice_log('contract_repair', {'turn': turn,
                        'kind': 'advance_next_dawn_input', 'advanced': gaps,
                        'next_farm_slots_before': hires + lands + len(gaps)})
                else:
                    self._choice_log('contract_repair', {'turn': turn,
                        'kind': 'advance_unaffordable_or_capacity', 'advanced': gaps,
                        'cost': cost, 'affordable': affordable, 'room': room})
        return super()._schedule_current_first_market(obs, projected, hard_buys, staffing,
            future_buys, stale_buys, sales, future_staffing, turn=turn, day=day, hour=hour,
            protected_cash_after_hard=protected_cash_after_hard)
