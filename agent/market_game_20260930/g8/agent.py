"""Public supply/demand allocation of compatible milk/wool production blocks."""
from collections import defaultdict
import math

from market_game_20260930.g5.agent import RankedCropAgent,factory as previous_factory
from market_game_20260930.g8.pastures import PastureTapes,certificates
from market_game_20260930.g2.seller import TimedFlowSeller
from market_game_20260930.ledger import known_demand,engine
from runtime.seller import sale_revenue
from runtime.view import own_farm,rival_farm,step_of

GOODS=('MILK','WOOL')


def future_deliveries(tapes,now,extra=None):
    selected=set(tapes.selected_animals)
    if extra:
        selected.add(extra)
    flow=defaultdict(lambda:defaultdict(int))
    for slot in certificates()['slots']:
        chosen=slot['id'] in selected
        product=slot['alternative' if chosen else 'original']['product']
        key='alternative_quantity' if chosen else 'original_quantity'
        for arrival in slot['deliveries']:
            if arrival['step'] is not None and now<=arrival['step']<=718:
                flow[arrival['step']][product]+=arrival[key]
    return flow


def rival_daily_supply(obs,forecasts):
    """Visible assets, not hidden opponent stock or its future plan."""
    visible=defaultdict(list)
    day=step_of(obs)//24
    for row in rival_farm(obs).get('tiles') or []:
        for tile in row:
            if not isinstance(tile,dict) or tile.get('animal') not in ('COW','SHEEP'):
                continue
            rule=engine.ANIMALS[tile['animal']]
            pending=min(2,max(1,int(tile.get('pending_care_bonus') or 0)))
            first=int(tile['placed_day'])+rule['first_yield_day']
            visible[rule['product']].append((first,rule['interval'],1+pending))
    result={}
    for d in range(day,30):
        result[d]={}
        for p in GOODS:
            capacity=sum(q/interval for first,interval,q in visible[p] if d>=first)
            # Actual recent public flow is helpful only while extrapolating a
            # near-term horizon; far out, visible mature capacity is the anchor.
            recent=sum(forecasts.get(p,[]))
            weight=.5*math.exp(-(d-day)/3)
            result[d][p]=(1-weight)*capacity+weight*recent
    return result


def portfolio_value(obs,tapes,forecasts,extra=None):
    now=step_of(obs)
    flow=future_deliveries(tapes,now,extra)
    for p in GOODS:
        flow[now][p]+=max(0,int((obs.get('private') or {}).get('shed',{}).get(p,0)))
    supply=rival_daily_supply(obs,forecasts)
    params=engine._resolve_market_params(obs['market'].get('params'))
    shops=(obs.get('town') or {}).get('unlocked_shops') or []
    current={p:int(obs['market']['inventory'][p]) for p in GOODS}
    projected=dict(current); fractional={p:0. for p in GOODS}
    spot_total=forward_total=0.
    for t in range(now,719):
        for p in GOODS:
            quantity=int(flow.get(t,{}).get(p,0))
            if quantity:
                spot_total+=sale_revenue(p,quantity,current[p],params)[0]
                revenue,cursor=sale_revenue(p,quantity,projected[p],params)
                forward_total+=revenue
                # Plan-wide own supply matters: each future block does not
                # get to sell at a price that ignores our other production.
                projected[p]=cursor
            fractional[p]+=supply[t//24][p]/24
            incoming=int(fractional[p]); fractional[p]-=incoming
            _,projected[p]=sale_revenue(p,incoming,projected[p],params)
            projected[p]-=known_demand(shops,p,t)
    # Unknown future shops and rival investments are not filled from a replay.
    # Shrink the long extrapolation toward today's public opportunity value.
    return .5*spot_total+.5*forward_total,dict(spot=spot_total,forward=forward_total,
                                            visible_rival_daily=supply)


class PastureAgent(RankedCropAgent):
    def __init__(self,mode='public',*,tree=None,cards=None):
        super().__init__('sync',tree=tree,cards=cards)
        if mode not in ('noop','cow','sheep','public'):
            raise ValueError(mode)
        self.pasture_mode=mode
        self.tapes=PastureTapes(self.tapes.original,self.tapes.loc)
        self.pasture_pool=defaultdict(list)
        for slot in certificates()['slots']:
            if slot['eligible']:
                self.pasture_pool[slot['decision_day']].append(slot)
        self.pasture_decisions=[]

    def _dawn(self,obs,day):
        slots=self.pasture_pool.get(day,[])
        if slots and self.pasture_mode!='noop':
            self.market_seller.update(obs)
            recovering=bool(self.deferred_cash_jobs or self.deferred_weed_jobs or self.pending_weed_jobs
                            or self.extra_hand_plan or self.weed_plan)
            safety='cash_below_2000' if float(own_farm(obs).get('money') or 0)<2000 else 'unresolved_repair' if recovering else None
            remaining=list(slots); changed=0
            while remaining:
                baseline,base_detail=portfolio_value(obs,self.tapes,self.market_seller.flow_prediction)
                choices=[]
                for slot in remaining:
                    value,detail=portfolio_value(obs,self.tapes,self.market_seller.flow_prediction,slot['id'])
                    cost=engine.ANIMALS[slot['alternative']['animal']]['cost']-engine.ANIMALS[slot['animal']]['cost']
                    advantage=value-baseline-cost
                    choices.append((advantage,slot,detail,cost))
                advantage,slot,detail,cost=max(choices,key=lambda x:x[0])
                selected=(self.pasture_mode=='public' and safety is None and advantage>=150 and changed<2)
                if self.pasture_mode in ('cow','sheep'):
                    selected=slot['alternative']['animal']==self.pasture_mode.upper()
                self.pasture_decisions.append(dict(id=slot['id'],day=day,xy=slot['xy'],
                    original=slot['animal'],alternative=slot['alternative']['animal'],
                    selected=slot['alternative']['animal'] if selected else slot['animal'],
                    changed=selected,advantage=advantage,cost_difference=cost,safety=safety,
                    original_units=slot['original']['total'],alternative_units=slot['alternative']['total'],
                    baseline_value=baseline,candidate_value=advantage+baseline+cost,
                    current_prices={p:obs['market']['prices'][p] for p in GOODS},
                    known_shops=list((obs.get('town') or {}).get('unlocked_shops') or [])))
                if selected:
                    self.tapes.choose_animal(slot); changed+=1
                remaining.remove(slot)
        super()._dawn(obs,day)


def factory(name):
    from market_game_20260930.players import RecordedMarketAgent
    modes={'ANIMAL_NOOP':'noop','ALL_COW':'cow','ALL_SHEEP':'sheep','PASTURE':'public'}
    if name not in modes:
        return previous_factory(name)
    return lambda:RecordedMarketAgent(PastureAgent(modes[name]),TimedFlowSeller(True))
