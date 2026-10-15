"""One causal public feature implementation shared by replay fitting and play."""
from __future__ import annotations
from collections import deque
import math

from market_game_20260930.competitive_seller import LedgerSeller
from market_game_20260930.ledger import PRODUCTS, engine
from runtime.seller import shop_demand
from runtime.view import rival_farm, step_of

NAMES = ('season','hour','next_shop_tick','normalized_inventory','relative_price',
         'relative_price_impact','daily_known_demand','rival_cash','active_tiles',
         'held_yield','visible_three_day_supply','backlog_belief','net_flow_ewma',
         'recent_24_supply','time_since_supply','flow_observable')
BINS = ((0,1),(1,4),(4,12),(12,24))


class PublicTracker(LedgerSeller):
    def __init__(self):
        super().__init__()
        self.history={p:deque(maxlen=24) for p in PRODUCTS}
        self.last_sale={p:-1000 for p in PRODUCTS}
        self.feature_step=None

    def update(self,obs):
        step=step_of(obs)
        if self.feature_step==step:
            return
        if self.feature_step is not None and step<self.feature_step:
            self.history={p:deque(maxlen=24) for p in PRODUCTS}
            self.last_sale={p:-1000 for p in PRODUCTS}
        super().update(obs)
        for p in PRODUCTS:
            receipt=self.ledger.last_records.get(p,{})
            net=receipt.get('rival_priced_net_supply',0)
            self.history[p].append(max(0,net))
            if net>0:
                self.last_sale[p]=step-1
        self.feature_step=step


def public_features(obs,tracker):
    step=step_of(obs)
    farm=rival_farm(obs)
    held={p:0. for p in PRODUCTS}
    active={p:0. for p in PRODUCTS}
    for row in farm.get('tiles') or []:
        for tile in row or []:
            if not isinstance(tile,dict):
                continue
            crop=tile.get('crop')
            animal=engine.ANIMALS.get(tile.get('animal'),{})
            p=crop or animal.get('product')
            if p in held:
                active[p]+=1
                held[p]+=max(0,int(tile.get('yield_units') or 0))
            if animal:
                active['FERTILIZER']+=1
                held['FERTILIZER']+=int(bool(tile.get('fertilizer_available')))
    visible=tracker._visible_supply(farm,int(obs.get('day') or 0),3)
    params=engine._resolve_market_params(obs['market'].get('params'))
    shops=obs.get('town',{}).get('unlocked_shops') or []
    rows=[]
    for p in PRODUCTS:
        par=params[p]; inv=int(obs['market']['inventory'][p]); base=max(1,float(par['base']))
        price=engine.market_price(p,inv,params)
        rec=tracker.ledger.last_records.get(p,{})
        rows.append([step/718,(step%24)/23,((4-step%4)%4)/4,
            (inv-par['I0'])/max(1,par['T']),price/base,
            (price-engine.market_price(p,inv+5,params))/base,
            sum(shop_demand(shops,p,t) for t in range(step,step+24))/24,
            math.log1p(max(0,float(farm.get('money') or 0)))/12,
            active[p]/100,held[p]/50,visible.get(p,0)/100,
            tracker.rival_backlog[p]/100,tracker.opponent_pressure.get(p,0)/10,
            sum(tracker.history[p])/100,min(48,step-tracker.last_sale[p])/48,
            float('rival_priced_net_supply' in rec)])
    assert all(len(x)==len(NAMES) for x in rows)
    return rows
