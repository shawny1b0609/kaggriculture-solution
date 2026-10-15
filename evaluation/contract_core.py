"""Audited production-contract primitives extracted from the historical evaluator.

Only imports were adapted. See research/original_evaluation for provenance.
"""
from collections import Counter, defaultdict
import copy
from kaggle_environments.envs.kaggriculture import kaggriculture as native
from evaluation import scenario_engine as rules
from tools.reliable_market_seller import ReliableMarketSeller

def clean(counter):
    return {str(k): int(v) for k, v in counter.items() if int(v)}


class SourceTransactions:
    """Observe successful source transactions, including partial order fills."""
    def __init__(self, state):
        self.farms = {id(f): s for s, f in enumerate(state[0].observation.farms)}
        self.orders = {id(o): (s, i) for s, st in enumerate(state)
                       for i, o in enumerate(st.action.get('market', [])[:10])}
        self.active = {}
        self.fills = defaultdict(int)
        self.costs = Counter()
        self.originals = {}

    def __enter__(self):
        for name in ('_parse_order', '_commit_unit', '_do_hire', '_do_buy_land'):
            self.originals[name] = getattr(rules, name)
        def parse(order):
            if id(order) in self.orders:
                s, i = self.orders[id(order)]
                self.active[s] = i
            return self.originals['_parse_order'](order)
        def unit(op, item, price, farm, *args):
            before = farm['money']
            ok = self.originals['_commit_unit'](op, item, price, farm, *args)
            s = self.farms[id(farm)]
            if ok:
                self.fills[s, self.active[s]] += 1
                if op != 'SELL':
                    self.costs[s] += before - farm['money']
            return ok
        def atomic(name, farm, *args):
            before = farm['money']
            result = self.originals[name](farm, *args)
            if farm['money'] != before:
                s = self.farms[id(farm)]
                self.fills[s, self.active[s]] += 1
                self.costs[s] += before - farm['money']
            return result
        rules._parse_order = parse
        rules._commit_unit = unit
        rules._do_hire = lambda f, *a: atomic('_do_hire', f, *a)
        rules._do_buy_land = lambda f, *a: atomic('_do_buy_land', f, *a)
        return self

    def __exit__(self, *args):
        for k, v in self.originals.items():
            setattr(rules, k, v)


class FixedSupply:
    def __init__(self, data, env):
        self.data, self.env = data, env
        self.seat = data['scenario']['winner_seat']
        self.reserve = Counter()
        self.seller = ReliableMarketSeller()
        self.debt = self.borrowed = self.peak_debt = 0.
        self.records = [dict(sales=0., costs=0., sold=Counter(), bought=Counter()) for _ in range(2)]
        self.losses = Counter()
        self.step = 0
        self.originals = {}
        self.buy_queue = []
        self.active_state = None
        self.initial_money = [float(f['money']) for f in env.state[0].observation.farms]

    @property
    def farm(self):
        return (self.active_state if self.active_state is not None else self.env.state)[0].observation.farms[self.seat]

    @property
    def private(self):
        return (self.active_state if self.active_state is not None else self.env.state)[self.seat].observation.private

    def replace_farm(self, source):
        money = self.farm['money']
        self.farm.clear()
        self.farm.update(copy.deepcopy(source))
        self.farm['money'] = money
        self.private['inventories'] = [{} for _ in range(1+len(source['hands']))]

    def deposit(self, private, reserve, e, losses):
        p, q = e['product'], e['quantity']
        shed = private['shed']
        if q < 0:
            assert reserve[p] >= -q and shed.get(p, 0) >= -q, (self.step, p, q, reserve[p], shed.get(p, 0))
            reserve[p] += q
            shed[p] += q
            return
        reserved = e['reserved']
        capacity = int(self.env.configuration.shedCapacity)
        # Guaranteed-production assumption: protect production inputs. If a
        # slower seller fills the shed, surplus goods, never inputs, are lost.
        while sum(shed.values()) + reserved > capacity:
            free = [k for k, n in shed.items() if n > reserve[k]]
            if not free:
                raise ValueError('production reservation exceeds capacity')
            chosen = min(free, key=lambda k: (native.MARKET_PARAMS.get(k, {}).get('base', 0), k))
            shed[chosen] -= 1
            losses[chosen] += 1
        free_room = max(0, capacity - sum(shed.values()) - reserved)
        taken = reserved + min(q-reserved, free_room)
        shed[p] = shed.get(p, 0) + taken
        reserve[p] += reserved
        losses[p] += q-taken

    def field(self, private, reserve, losses):
        row = self.data['rows'][self.step]
        for e in row['field']:
            self.deposit(private, reserve, e, losses)
        for p, q in row['seed_delta'].items():
            private['seeds'][p] = private['seeds'].get(p, 0) + q
            assert private['seeds'][p] >= 0

    def action(self, obs):
        row = self.data['rows'][self.step]
        projected, reserved = copy.deepcopy(self.private), copy.deepcopy(self.reserve)
        self.field(projected, reserved, Counter())
        orders = [item['order'] for item in row['operations']]
        assert len(orders) <= 10
        forecast = Counter()
        for future in self.data['rows'][self.step:min(719, (self.step//24+1)*24)]:
            for event in future['field'] + future['night']:
                if event['quantity'] > 0:
                    forecast[event['product']] += event['quantity'] - event['reserved']
        # Use live prices for the immediate operations. This is a liquidity
        # budget, not a future quotation or source final-money target.
        budget, hires = 0., int(row['field_farm']['hires_today'])
        lands = len(row['field_farm']['unlocked_quadrants']) - 1
        for order in orders:
            op = order[0]
            if op == 'HIRE':
                budget += native._hire_cost(hires, self.env.configuration.farmHandCostMult)
                hires += 1
            elif op == 'BUY_LAND':
                budget += native.LAND_PRICES[lands]
                lands += 1
            elif op == 'BUY_SEED':
                budget += native.CROPS[order[1]]['seed'] * order[2]
            elif op == 'BUY_ANIMAL':
                budget += native.ANIMALS[order[1]]['cost'] * order[2]
            elif op == 'BUY_PRODUCT':
                p, n = order[1:]
                budget += sum(native.market_price(p, obs['market']['inventory'][p]-j-1,
                                                    obs['market'].get('params')) for j in range(n))
        seller_obs = copy.deepcopy(obs)
        seller_obs['farms'][self.seat]['money'] -= self.debt
        plan = self.seller.plan(seller_obs, dict(projected['shed']), dict(reserved),
                               minimum_cash=budget, available_order_slots=10-len(orders),
                               required_capacity=sum(item['flow']['quantity'] for item in row['operations']
                                                     if item['flow']),
                               projected_arrivals=dict(forecast), terminal=self.step>=718)
        return dict(farmer=['PASS'], hands=[], market=plan.orders + orders)

    def borrow_for(self, farm, amount):
        shortage = max(0., amount - farm['money'])
        farm['money'] += shortage
        self.debt += shortage
        self.borrowed += shortage
        self.peak_debt = max(self.peak_debt, self.debt)

    def __enter__(self):
        for key in ('_process_market', '_commit_unit', '_do_hire', '_do_buy_land', '_end_of_day'):
            self.originals[key] = getattr(rules, key)
        def market(state, env):
            # Kaggle passes a copied working state into the interpreter;
            # env.state still denotes the previous frame until env.step ends.
            self.active_state = state
            row = self.data['rows'][self.step]
            self.replace_farm(row['field_farm'])
            self.field(self.private, self.reserve, self.losses)
            self.buy_queue = [[item['order'][1], item['flow']['quantity'], item['flow']['reserved']]
                              for item in row['operations'] if item['flow']]
            self.farm_ids = {id(f): s for s, f in enumerate(state[0].observation.farms)}
            result = self.originals['_process_market'](state, env)
            assert not self.buy_queue, ('incomplete fixed procurement', self.step, self.buy_queue)
            return result
        def commit(op, item, price, farm, private, market, cap=100):
            seat = self.farm_ids[id(farm)]
            protected = 0
            if seat == self.seat and op != 'SELL':
                self.borrow_for(farm, price)
                if op in ('BUY_PRODUCT', 'BUY_ANIMAL'):
                    assert self.buy_queue and self.buy_queue[0][0] == item
                    protected = int(self.buy_queue[0][2] > 0)
                    # Make room only for a guaranteed production purchase;
                    # the displaced sale stock is explicitly recorded as loss.
                    while sum(private['shed'].values()) >= cap:
                        free = [p for p, n in private['shed'].items() if n > self.reserve[p]]
                        if not free:
                            raise ValueError('purchase cannot fit protected warehouse')
                        p = min(free, key=lambda p: (native.MARKET_PARAMS.get(p, {}).get('base', 0), p))
                        private['shed'][p] -= 1
                        self.losses[p] += 1
            before = float(farm['money'])
            ok = self.originals['_commit_unit'](op, item, price, farm, private, market, cap)
            record = self.records[seat]
            if ok:
                if op == 'SELL':
                    record['sales'] += float(farm['money']) - before
                    record['sold'][item] += 1
                    if seat == self.seat:
                        repayment = min(self.debt, farm['money'])
                        farm['money'] -= repayment
                        self.debt -= repayment
                else:
                    record['costs'] += before - float(farm['money'])
                    record['bought'][op+':'+item] += 1
                    if seat == self.seat and op in ('BUY_PRODUCT', 'BUY_ANIMAL'):
                        self.reserve[item] += protected
                        self.buy_queue[0][1] -= 1
                        self.buy_queue[0][2] -= protected
                        if not self.buy_queue[0][1]:
                            self.buy_queue.pop(0)
            elif seat == self.seat and op != 'SELL':
                raise ValueError(('guaranteed purchase rejected', self.step, op, item))
            return ok
        def atomic(key, farm, *args):
            seat = self.farm_ids[id(farm)]
            if seat == self.seat:
                cost = (native._hire_cost(farm['hires_today'], args[-1]) if key == '_do_hire'
                        else native.LAND_PRICES[len(farm['unlocked_quadrants'])-1])
                self.borrow_for(farm, cost)
            before = float(farm['money'])
            value = self.originals[key](farm, *args)
            self.records[seat]['costs'] += before - float(farm['money'])
            return value
        def end_day(state, env, day):
            result = self.originals['_end_of_day'](state, env, day)
            for e in self.data['rows'][self.step]['night']:
                self.deposit(self.private, self.reserve, e, self.losses)
            return result
        rules._process_market = market
        rules._commit_unit = commit
        rules._do_hire = lambda f, *a: atomic('_do_hire', f, *a)
        rules._do_buy_land = lambda f, *a: atomic('_do_buy_land', f, *a)
        rules._end_of_day = end_day
        self.replace_farm(self.data['rows'][0]['pre_farm'])
        self.private['seeds'] = copy.deepcopy(self.data['initial_seeds'])
        for e in self.data['initial']:
            self.deposit(self.private, self.reserve, e, self.losses)
        return self

    def __exit__(self, *args):
        for k, v in self.originals.items():
            setattr(rules, k, v)
