"""Read-only official-rule certificates for wheat-to-carrot crop cycles.

This is NOT a deployable crop adapter and makes no profit claim. It checks
one tile with sufficient seeds/fertilizer. Procurement, feed, delivery and
whole-farm reference-state updates remain integration obligations.
"""
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

from common import AGENT
from market_game_20260930.ledger import engine

OUT = Path(__file__).resolve().parent
ASSET = AGENT / 'asset.json.gz'
FIELD = {'PLANT', 'WATER', 'FERTILIZE', 'HARVEST', 'DIG'}


def simulate_cycle(events, crop):
    """Field actions -> hourly decay -> midnight, exactly the engine order.

    Tapes already index pre-action observations (episode action delay has
    already been resolved by the source compiler); do not shift them again.
    Worker identity/movement is irrelevant only for this single-tile check.
    """
    first, last = events[0]['step'], events[-1]['step']
    farm = dict(tiles=[[None for _ in range(10)] for _ in range(10)],
                farmer=[0, 0], hands=[], unlocked_quadrants=[0])
    private = dict(seeds={crop: 1}, shed={}, inventories=[{'FERTILIZER': 1000}])
    schedule = defaultdict(list)
    for event in events:
        schedule[event['step']].append(event)
    dawns = {}
    crop_losses = []
    harvested = 0
    costs = Counter()
    for step in range(first, last + 1):
        day = step // 24
        if step % 24 == 0:
            dawns[str(day)] = copy.deepcopy(farm['tiles'][0][0])
        for event in schedule[step]:
            command = list(event['action'])
            if command[0] == 'PLANT':
                command[1] = crop
            before_yield = private['inventories'][0].get(crop, 0)
            before_fertilizer = private['inventories'][0].get('FERTILIZER', 0)
            engine._apply_unit_action(farm, private, 0, command, 10, day, 24, 100)
            harvested += private['inventories'][0].get(crop, 0) - before_yield
            costs['fertilizer'] += before_fertilizer - private['inventories'][0].get('FERTILIZER', 0)
            costs['water_actions'] += int(command[0] == 'WATER')
        engine._decay_plants(farm, step)
        if (step + 1) % 24 == 0:
            engine._daily_refresh_plants(farm, day, 24)
        tile = farm['tiles'][0][0]
        if isinstance(tile, dict) and tile.get('kind') == 'WEED' and not crop_losses:
            crop_losses.append(step)
    return dict(harvested=harvested, cleared=farm['tiles'][0][0] is None,
                crop_losses=crop_losses, costs=dict(costs), dawns=dawns)


def extract_cycles(asset):
    route = asset['routes'][0]
    assert route['path_index'] == 0
    growing, cycles, daily = {}, [], {}
    for day in range(30):
        loc = route['edges'][min(day, 28)]['example']
        rows = asset['sources'][loc]['tapes'].get(str(day), [])
        daily[day] = rows
        for hour, row in enumerate(rows):
            for unit, (command, pos) in enumerate(zip([row['farmer'], *row['hands']], row['positions'])):
                xy = tuple(pos)
                event = dict(step=day*24 + hour, unit=unit, action=list(command))
                if command[0] == 'PLANT':
                    growing.pop(xy, None)
                    if command[1] == 'WHEAT':
                        growing[xy] = dict(xy=list(xy), plant_day=day,
                                          plant_hour=hour, events=[])
                cycle = growing.get(xy)
                if cycle is None:
                    continue
                if command[0] in FIELD:
                    cycle['events'].append(event)
                if command[0] in ('HARVEST', 'DIG'):
                    growing.pop(xy)
                    if command[0] == 'HARVEST':
                        cycle.update(harvest_day=day, harvest_hour=hour, harvest_unit=unit,
                                     age=day-cycle['plant_day'])
                        cycles.append(cycle)
    for cycle in cycles:
        day, hour, unit = cycle['harvest_day'], cycle['harvest_hour'], cycle['harvest_unit']
        dependencies = []
        delivery = 'midnight_auto_drop' if day < 29 else 'terminal_bag_risk'
        for h, row in enumerate(daily[day][hour + 1:], hour + 1):
            commands = [row['farmer'], *row['hands']]
            command = commands[unit] if unit < len(commands) else ['PASS']
            if command[0] == 'DROP':
                delivery = 'DROP'
                break
            if (command[0] == 'FEED' or
                (command[0] in ('PLACE', 'PICKUP') and len(command) > 1 and command[1] == 'WHEAT')):
                dependencies.append(dict(step=day*24+h, command=command))
        cycle['before_delivery_wheat_dependencies'] = dependencies
        cycle['delivery'] = delivery
    return cycles


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/local/crop_certificates.json'))
    output=parser.parse_args().out
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with gzip.open(ASSET, 'rt', encoding='utf-8') as handle:
        asset = json.load(handle)
    cycles = extract_cycles(asset)
    for cycle in cycles:
        wheat = simulate_cycle(cycle['events'], 'WHEAT')
        carrot = simulate_cycle(cycle['events'], 'CARROT')
        cycle['wheat'] = wheat
        cycle['carrot'] = carrot
        cycle['field_compatible'] = bool(wheat['cleared'] and carrot['cleared'] and
            wheat['harvested'] > 0 and carrot['harvested'] > 0 and not carrot['crop_losses'])
        cycle['simpler_delivery_candidate'] = bool(cycle['field_compatible'] and
            not cycle['before_delivery_wheat_dependencies'] and cycle['delivery'] != 'terminal_bag_risk')
    compatible = [c for c in cycles if c['field_compatible']]
    simple = [c for c in cycles if c['simpler_delivery_candidate']]
    report = dict(version='single_tile_crop_certificate_v1',
                  status='diagnostic_only_not_runtime_or_profit_validation',
                  identity={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ASSET, Path(engine.__file__), Path(__file__))},
                  summary=dict(wheat_cycles=len(cycles),
                               age_counts=dict(Counter(str(c['age']) for c in cycles)),
                               field_compatible=len(compatible),
                               compatible_age_counts=dict(Counter(str(c['age']) for c in compatible)),
                               simpler_delivery_candidates=len(simple),
                               simpler_candidates_from_day10=sum(c['plant_day'] >= 10 for c in simple),
                               baseline_failed_cycles=sum(not c['wheat']['cleared'] or
                                                          c['wheat']['harvested'] <= 0 for c in cycles)),
                  cycles=cycles)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
