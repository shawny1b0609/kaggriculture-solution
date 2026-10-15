"""Online own-flow reconciliation using only legal observations and own actions."""
from __future__ import annotations

from collections import Counter
import copy

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

PRODUCTS = tuple(engine.PRODUCTS)


def product_totals(private):
    result = Counter(private.get('shed') or {})
    for bag in private.get('inventories') or []:
        result.update(bag or {})
    return {p: int(result.get(p, 0)) for p in PRODUCTS}


def field_private(obs, action, configuration=None):
    """Execute only our known farm commands, including atomic PLANT validation."""
    cfg = configuration or {}
    farm = copy.deepcopy(obs['farms'][int(obs['player'])])
    private = copy.deepcopy(obs.get('private') or {})
    actions = [action.get('farmer') or ['PASS'], *(action.get('hands') or [])]
    seed_demand = Counter(a[1] for a in actions
                          if isinstance(a, list) and len(a) >= 2 and a[0] == 'PLANT')
    blocked = {p for p, q in seed_demand.items() if q > (private.get('seeds') or {}).get(p, 0)}
    for i, command in enumerate(actions):
        if isinstance(command, list) and len(command) >= 2 and command[0] == 'PLANT' and command[1] in blocked:
            command = ['PASS']
        engine._apply_unit_action(farm, private, i, command,
                                 int(cfg.get('boardSize', len(farm.get('tiles') or []) or 10)),
                                 int(obs.get('day') or 0), int(cfg.get('turnsPerDay', 24)),
                                 int(cfg.get('shedCapacity', 100)))
    return private


def known_demand(shops, product, step, configuration=None):
    cfg = configuration or {}
    q = 0
    if step % max(1, int(cfg.get('townShopSellInterval', 4))) == 0:
        for shop in shops:
            goods = engine.SHOPS.get(shop, [])
            if product in goods:
                q += 2 if len(goods) == 1 else 1
    if step % max(1, int(cfg.get('townCenterSellInterval', 24))) == 0 and product in engine.TOWN_CENTER_PRODUCTS:
        q += 1
    return q


class OwnFlowLedger:
    def __init__(self):
        self.pending = None
        self.stats = Counter()
        self.last_records = {}

    def record(self, obs, action, configuration=None):
        cfg = dict(configuration or {})
        step = int(obs.get('step', int(obs.get('day', 0)) * 24 + int(obs.get('hour', 0))))
        private = field_private(obs, action, cfg)
        orders = copy.deepcopy((action.get('market') or [])[:int(cfg.get('maxMarketOrdersPerTurn', 10))])
        self.pending = dict(step=step, totals=product_totals(private),
                            inventory=dict(obs['market']['inventory']),
                            params=engine._resolve_market_params(obs['market'].get('params')),
                            shops=list((obs.get('town') or {}).get('unlocked_shops') or []),
                            orders=orders, config=cfg)

    def observe(self, obs):
        row = self.pending
        step = int(obs.get('step', int(obs.get('day', 0)) * 24 + int(obs.get('hour', 0))))
        self.last_records = {}
        if row is None:
            return {}
        if step == row['step']:
            return {}  # Repeated inspection must not consume the receipt.
        self.pending = None
        if step != row['step'] + 1:
            self.stats['skipped_transition'] += 1
            return {}
        totals = product_totals(obs.get('private') or {})
        cfg = row['config']
        night = step % int(cfg.get('turnsPerDay', 24)) == 0
        full_shed = sum((obs.get('private', {}).get('shed') or {}).values()) >= int(cfg.get('shedCapacity', 100))
        inventory = obs['market']['inventory']
        signed = {}
        for p in PRODUCTS:
            own = row['totals'][p] - totals[p]
            demand = known_demand(row['shops'], p, row['step'], cfg)
            before, after_clearing = row['inventory'][p], int(inventory[p]) + demand
            sales = sum(max(0, int(o[2])) for o in row['orders']
                        if len(o) >= 3 and o[0] == 'SELL' and o[1] == p)
            # Endpoint floor tests suffice for one-way products. For the two
            # buyable goods use a conservative possible same-turn excursion.
            possible_high = max(before, after_clearing)
            if p in ('WHEAT', 'FERTILIZER') and sales:
                possible_high = max(possible_high, before + sales + 100)
            floor_ambiguous = sales > 0 and engine.market_price(p, possible_high, row['params']) == 1
            unknown = ('night_overflow' if night and full_shed else
                       'floor_sales' if floor_ambiguous else None)
            entry = dict(own_net_sold=own, consumption=demand, unknown=unknown)
            self.last_records[p] = entry
            if unknown:
                self.stats[unknown] += 1
                continue
            net = int(inventory[p]) - before - own + demand
            signed[p] = net
            entry['rival_priced_net_supply'] = net
            self.stats['reconciled_product_steps'] += 1
            if own < 0:
                self.stats['own_net_buy_product_steps'] += 1
        return signed


def visible_harvest(previous_farm, farm):
    """Observable yield disappearing from tiles, not confirmed shed arrivals."""
    harvested = Counter()
    for y, row in enumerate(previous_farm.get('tiles') or []):
        for x, old in enumerate(row):
            if not isinstance(old, dict):
                continue
            rows = farm.get('tiles') or []
            new = rows[y][x] if y < len(rows) and x < len(rows[y]) else None
            new = new if isinstance(new, dict) else {}
            product = old.get('crop') or engine.ANIMALS.get(old.get('animal'), {}).get('product')
            same = (old.get('crop'), old.get('animal'), old.get('placed_day'), old.get('planted_day')) == (
                new.get('crop'), new.get('animal'), new.get('placed_day'), new.get('planted_day'))
            held = int(old.get('yield_units') or 0)
            if product:
                harvested[product] += max(0, held - (int(new.get('yield_units') or 0) if same else 0))
            if same and old.get('animal') and old.get('fertilizer_available') and not new.get('fertilizer_available'):
                harvested['FERTILIZER'] += 1
    return harvested
