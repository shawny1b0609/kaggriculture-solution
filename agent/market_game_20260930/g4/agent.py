"""Market-responsive short crop cycles on the immutable 731291 farm skeleton."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOCAL_IMPLEMENTATION = ROOT/'featured_forest_20260925/experiments/compact_switch_executor_20260929'
if (LOCAL_IMPLEMENTATION/'compact_agent.py').is_file():
    sys.path.insert(0, str(LOCAL_IMPLEMENTATION))
from market_game_20260930.g2.seller import TimedFlowSeller
from market_game_20260930.ledger import known_demand, engine
from market_game_20260930.g4.cycles import CycleTapes, cycle_pool
from compact_agent import CompactRuleAgent, make_cards
from tools.portable_tree_agent import PortableTree
from runtime.seller import sale_revenue
from runtime.view import step_of

ASSET = ROOT/'featured_forest_20260925/experiments/production_tree_search_20260929/source_packages/strong731_single_v1/assets/asset.json.gz'


@lru_cache(maxsize=1)
def source_tree():
    tree = PortableTree(ASSET)
    return tree, make_cards(tree)


def cycle_value(obs, cycle, forecasts):
    """Conservative public opportunity score, NOT a future-price oracle.

    Blend spot proceeds with a public-flow/known-shop projection. The rival
    forecast is shrunk in influence over the longer 2-3 day crop horizon.
    Both crops pay their seed costs; wheat's lost units count at their full
    market value, so replacing feed production is not treated as free.
    """
    params = engine._resolve_market_params(obs['market'].get('params'))
    now = step_of(obs)
    delivery = ((cycle['harvest_day']+1)*24 if cycle['delivery'] == 'midnight_auto_drop'
                else cycle['harvest_day']*24 + cycle['harvest_hour']+1)
    span = max(1, delivery-now)
    shops = obs.get('town', {}).get('unlocked_shops') or []
    scores, details = {}, {}
    for crop, key in (('WHEAT', 'wheat'), ('CARROT', 'carrot')):
        quantity = cycle[key]['harvested']
        current = int(obs['market']['inventory'][crop])
        demand = sum(known_demand(shops, crop, step) for step in range(now, min(719, now+span)))
        rival = sum(forecasts.get(crop, []))*span/24
        projected = current + round(rival) - demand
        spot_revenue = sale_revenue(crop, quantity, current, params)[0]
        forward_revenue = sale_revenue(crop, quantity, projected, params)[0]
        value = .75*spot_revenue + .25*forward_revenue - engine.CROPS[crop]['seed']
        scores[crop] = value
        details[crop] = dict(quantity=quantity, inventory=current, known_demand=demand,
                            rival_quantity_estimate=rival, spot_value=spot_revenue,
                            projected_value=forward_revenue, net_value=value)
    return scores['CARROT']-scores['WHEAT'], details


class FlexibleCropAgent(CompactRuleAgent):
    def __init__(self, mode='public', limit=8, per_day=1, *, tree=None, cards=None):
        if tree is None:
            tree, cards = source_tree()
        super().__init__(tree, mode='rule', cards=cards)
        if mode not in ('wheat', 'carrot', 'public'):
            raise ValueError(mode)
        locs = {e['example'] for e in tree.routes[0]['edges']}
        assert len(locs) == 1 and len(tree.routes) == 1
        self.tapes = CycleTapes(tree.tapes, next(iter(locs)))
        self.crop_mode = mode
        self.crop_pool = defaultdict(list)
        for cycle in cycle_pool(limit, per_day):
            self.crop_pool[cycle['plant_day']].append(cycle)
        self.crop_decisions = []

    def _dawn(self, obs, day):
        for cycle in self.crop_pool.get(day, []):
            selected = self.crop_mode == 'carrot'
            advantage, details = None, {}
            safety = None
            if self.crop_mode == 'public':
                self.market_seller.update(obs)
                advantage, details = cycle_value(obs, cycle, self.market_seller.flow_prediction)
                cash = float(obs['farms'][int(obs['player'])]['money'])
                if cash < 2000:
                    safety = 'cash_below_2000'
                elif self.deferred_cash_jobs or self.deferred_weed_jobs or self.pending_weed_jobs:
                    safety = 'unresolved_execution_debt'
                selected = safety is None and advantage >= 20
            if selected:
                self.tapes.choose_carrot(cycle)
            self.crop_decisions.append(dict(id=cycle['id'], day=day, xy=cycle['xy'],
                                           selected='CARROT' if selected else 'WHEAT',
                                           advantage=advantage, details=details, safety=safety))
        super()._dawn(obs, day)


def factory(name):
    from market_game_20260930.g2.evaluate import factory as baseline_factory
    from market_game_20260930.players import RecordedMarketAgent
    variants = {'NOOP': ('wheat', 8, 1), 'CARROT8': ('carrot', 8, 1), 'FLEX8': ('public', 8, 1),
                'CARROT15': ('carrot', 15, 1), 'FLEX15': ('public', 15, 1)}
    if name not in variants:
        return baseline_factory(name)
    mode, limit, per_day = variants[name]
    return lambda: RecordedMarketAgent(FlexibleCropAgent(mode, limit, per_day), TimedFlowSeller(True))
