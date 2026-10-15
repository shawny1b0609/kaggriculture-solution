"""Public learned flow + multi-time execution, preserving all farm protections."""
from functools import lru_cache
import math

from market_game_20260930.competitive_seller import CompetitiveSeller,dual_sale
from market_game_20260930.g2.features import PublicTracker,public_features,PRODUCTS,BINS
from market_game_20260930.g2.model import load_model
from runtime.seller import sale_revenue,shop_demand,engine
from runtime.view import own_farm,step_of


def ordered_sale(item,ours,theirs,inventory,position,params=None,price_fn=None):
    """Receipts when our product order is before/same/after the rival's slot."""
    if position==0:
        return dual_sale(item,ours,theirs,inventory,params,price_fn)
    first_q,second_q=(ours,theirs) if position<0 else (theirs,ours)
    first,_,cursor=dual_sale(item,first_q,0,inventory,params,price_fn)
    second,_,cursor=dual_sale(item,second_q,0,cursor,params,price_fn)
    return (first,second,cursor) if position<0 else (second,first,cursor)


class TimedFlowSeller(CompetitiveSeller,PublicTracker):
    def __init__(self,learned=True):
        super().__init__(24,0.)
        self.learned=bool(learned)
        self.model=load_model() if learned else None
        self.flow_step=None
        self.flow_prediction={}

    def update(self,obs):
        step=step_of(obs)
        super().update(obs)
        if self.flow_step==step:
            return
        features=public_features(obs,self)
        self.flow_prediction={}
        for p,x in zip(PRODUCTS,features):
            if self.learned:
                pred=self.model.predict_one(x)
            else:
                pressure=x[12]*10;visible=x[10]*100
                pred=[max(0,pressure*(.75**a-.75**b)/.25+visible*(b-a)/72) for a,b in BINS]
            self.flow_prediction[p]=[max(0.,min(100.,float(v))) for v in pred]
        self.flow_step=step

    def _timed_quantity(self,obs,item,stock,projected_arrival,visible_supply):
        reference=super()._timed_quantity(obs,item,stock,projected_arrival,visible_supply)
        step=step_of(obs)
        # Do not silently change input-good arbitrage, bootstrap or end game.
        if (item in ('WHEAT','FERTILIZER') or stock<2 or step>695
                or float(own_farm(obs).get('money') or 0)<10000):
            return reference
        params=engine._resolve_market_params(obs['market'].get('params'))
        inventory=int(obs['market']['inventory'][item]);base=params[item]['base']
        shops=obs.get('town',{}).get('unlocked_shops') or []
        horizons=sorted({t+1-step for t in range(step,min(718,step+24)) if shop_demand(shops,item,t)>0})
        if not horizons:
            return reference
        candidates=sorted({0,stock,reference,*[math.ceil(stock*f) for f in (.25,.5,.75)]})
        forecast=self.flow_prediction[item]
        price=lru_cache(maxsize=None)(lambda inv:engine.market_price(item,inv,params))

        @lru_cache(maxsize=None)
        def sale(a,b,inv,pos):
            return ordered_sale(item,a,b,inv,pos,params,price)

        def cum(t):
            return sum(q*max(0,min(t,b)-a)/(b-a) for q,(a,b) in zip(forecast,BINS))

        @lru_cache(maxsize=None)
        def future_cursor(after_now,horizon,scale):
            cursor=after_now
            for t in range(1,horizon+1):
                cursor-=shop_demand(shops,item,step+t-1)
                r=max(0,round(scale*cum(t+1))-round(scale*cum(t))) if t<24 else round(scale*forecast[-1]/12)
                if t==horizon:
                    return cursor,r
                _,_,cursor=sale(0,r,cursor,0)
            raise AssertionError(horizon)

        best=(-float('inf'),reference,0)
        scores={}
        for q in candidates:
            for horizon in horizons:
                arrivals=int(max(0,projected_arrival)*min(1,horizon/max(1,24-step%24)))
                values=[];weights=[]
                for scale,sw in ((.5,.25),(1.,.5),(1.5,.25)):
                    r0=round(scale*forecast[0])
                    for position,pw in ((-1,.25),(0,.5),(1,.25)):
                        now,_,cursor=sale(q,r0,inventory,position)
                        inv,rh=future_cursor(cursor,horizon,scale)
                        later,_,_=sale(stock-q+arrivals,rh,inv,position)
                        values.append(now+later);weights.append(sw*pw)
                mean=sum(w*v for w,v in zip(weights,values))
                risk=math.sqrt(sum(w*max(0,mean-v)**2 for w,v in zip(weights,values)))
                score=mean-.2*risk-.002*horizon*(stock-q)*base
                scores[q]=max(scores.get(q,-float('inf')),score)
                if (score,q,-horizon)>(best[0],best[1],-best[2]):
                    best=score,q,horizon
        chosen=max(best[1],min(stock,self.overdue.get(item,0)))
        self.audit['flow_calls']+=1
        self.audit['flow_changed_vs_self24']+=int(chosen!=reference)
        self.audit['flow_more_units']+=max(0,chosen-reference)
        self.audit['flow_less_units']+=max(0,reference-chosen)
        self.timing_details[item]=dict(reason='LEARNED_FLOW_TIMING' if self.learned else 'RULE_FLOW_TIMING',
            quantity=chosen,reference_quantity=reference,horizon=best[2],forecast=forecast,scores=scores)
        return chosen
