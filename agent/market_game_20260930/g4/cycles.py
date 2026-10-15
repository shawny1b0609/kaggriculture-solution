"""Compile isolated crop-cycle choices without changing frozen runtime code."""
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES = ROOT/'market_game_20260930/g3/farm_probe/crop_cycle_certificates.json'


def add_amount(mapping, item, amount):
    mapping[item] = int(mapping.get(item, 0)) + int(amount)
    if mapping[item] == 0:
        mapping.pop(item)


@lru_cache(maxsize=1)
def available_cycles():
    bundled = Path(__file__).with_name('cycle_bank.json')
    data = json.loads((bundled if bundled.exists() else CERTIFICATES).read_text(encoding='utf-8'))
    result = []
    for source in data['cycles']:
        if (source['simpler_delivery_candidate'] and source['plant_day'] >= 10
                and source['age'] in (2, 3)):
            row = copy.deepcopy(source)
            row['id'] = f"d{row['plant_day']}_{row['xy'][0]}_{row['xy'][1]}"
            result.append(row)
    return tuple(sorted(result, key=lambda c: (c['plant_day'], c['plant_hour'], c['xy'])))


def cycle_pool(limit=8, per_day=1):
    result, counts = [], Counter()
    for row in available_cycles():
        if counts[row['plant_day']] < per_day and len(result) < limit:
            result.append(copy.deepcopy(row))
            counts[row['plant_day']] += 1
    return result


class CycleTapes:
    """Per-match copies of plans and exact expected dawn crop states.

    A selected cycle rejoins its original empty tile at the same HARVEST.
    We change the reference witness, never hide differences in live obs or
    disable production matching. Shared base assets stay read-only.
    """
    def __init__(self, original, loc):
        self.original = original
        self.loc = loc
        self.overrides = {}
        self.dawn_overrides = defaultdict(dict)
        self.selected = []

    def day_tape(self, loc, day):
        if (str(loc), int(day)) in self.overrides:
            return copy.deepcopy(self.overrides[str(loc), int(day)])
        return self.original.day_tape(loc, day)

    def dawn_farm(self, loc, day):
        result = self.original.dawn_farm(loc, day)
        if result is not None and str(loc) == self.loc:
            for (x, y), tile in self.dawn_overrides.get(int(day), {}).items():
                result['tiles'][y][x] = copy.deepcopy(tile)
        return result

    def choose_carrot(self, cycle):
        if not (cycle['simpler_delivery_candidate'] and cycle['age'] in (2, 3)
                and cycle['plant_day'] >= 10 and cycle['plant_hour'] >= 2):
            raise ValueError('uncertified crop cycle')
        if cycle['id'] in self.selected:
            raise ValueError('cycle selected twice')
        first, last = cycle['events'][0], cycle['events'][-1]
        pday, phour = divmod(first['step'], 24)
        hday, hhour = divmod(last['step'], 24)
        planting = self.day_tape(self.loc, pday)
        harvest = planting if pday == hday else self.day_tape(self.loc, hday)
        seed_step = planting[phour]
        commands = [seed_step['farmer'], *seed_step['hands']]
        assert commands[first['unit']] == ['PLANT', 'WHEAT']
        assert seed_step['positions'][first['unit']] == cycle['xy']
        # Verify canonical fields, not only a total yield or crop identifier.
        for day, expected in cycle['wheat']['dawns'].items():
            farm = self.original.dawn_farm(self.loc, int(day))
            x, y = cycle['xy']
            assert farm['tiles'][y][x] == expected, (cycle['id'], day, 'canonical dawn mismatch')
        seed_event = seed_step['unit_events'][first['unit']]['seeds']
        assert seed_event == {'WHEAT': -1}
        commands[first['unit']][:] = ['PLANT', 'CARROT']
        seed_event.clear()
        seed_event['CARROT'] = -1
        add_amount(seed_step['seed_delta_units'], 'WHEAT', 1)
        add_amount(seed_step['seed_delta_units'], 'CARROT', -1)
        wq, cq = cycle['wheat']['harvested'], cycle['carrot']['harvested']
        row = harvest[hhour]
        assert [row['farmer'], *row['hands']][last['unit']] == ['HARVEST']
        assert row['positions'][last['unit']] == cycle['xy']
        assert row['yielded'].get('WHEAT', 0) >= wq
        add_amount(row['yielded'], 'WHEAT', -wq)
        add_amount(row['yielded'], 'CARROT', cq)
        # Harvested cargo cannot feed animals or undergo named WHEAT transfers
        # before being deposited. Already enforced by the cycle certificate.
        deposited = False
        for step in harvest[hhour + 1:]:
            units = [step['farmer'], *step['hands']]
            unit = last['unit']
            command = units[unit] if unit < len(units) else ['PASS']
            if command == ['DROP']:
                event = step['unit_events'][unit]['shed']
                if event.get('WHEAT', 0) < wq:
                    raise ValueError('source partial DROP is not certified for substitution')
                add_amount(event, 'WHEAT', -wq)
                add_amount(event, 'CARROT', cq)
                add_amount(step['shed_delta_units'], 'WHEAT', -wq)
                add_amount(step['shed_delta_units'], 'CARROT', cq)
                deposited = True
                break
            if command[0] == 'FEED' or (len(command) > 1 and command[0] in ('PLACE', 'PICKUP') and command[1] == 'WHEAT'):
                raise ValueError('crop cargo has a wheat dependency')
        assert deposited or hday < 29, 'terminal harvest without a delivery'
        # Commit only after every validation succeeds (transactional compile).
        self.overrides[self.loc, pday] = planting
        self.overrides[self.loc, hday] = harvest
        for day, expected in cycle['carrot']['dawns'].items():
            self.dawn_overrides[int(day)][tuple(cycle['xy'])] = copy.deepcopy(expected)
        self.selected.append(cycle['id'])

    def total_yield(self):
        result = Counter()
        for day in range(30):
            for step in self.day_tape(self.loc, day):
                result.update(step['yielded'])
        return dict(result)
