import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import actor_input, load_entry
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as engine
from market_game_20260930.g2.features import PublicTracker, public_features, NAMES
from market_game_20260930.g2.model import FlowModel
from market_game_20260930.g2.seller import TimedFlowSeller, ordered_sale

def observation(step=0):
    env=make('kaggriculture',configuration={'seed':0},debug=True); env.reset()
    obs,_=actor_input(env,0)
    obs.update(step=step,day=step//24,hour=step%24)
    return obs

class ReleaseTests(unittest.TestCase):
    def test_frozen_runtime_bytes(self):
        manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            if row['path'].startswith('agent/'):
                self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'],row['path'])

    def test_public_features_only(self):
        obs=observation(); tracker=PublicTracker(); tracker.update(obs)
        before=public_features(obs,tracker)
        altered=copy.deepcopy(obs)
        altered.update(future_prices={'MILK':999999},opponent_private={'shed':{'MILK':9999}},episode_id='unseen')
        self.assertEqual(before,public_features(altered,tracker))
        self.assertEqual(len(NAMES),16)
        self.assertTrue(all(len(row)==16 for row in before))

    def test_portable_model(self):
        obs=observation(); tracker=PublicTracker(); tracker.update(obs)
        for features in public_features(obs,tracker):
            pred=FlowModel().predict_one(features)
            self.assertEqual(len(pred),4)
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in pred))

    def test_order_slots_and_terminal_liquidation(self):
        for slots in (0,1,2,10):
            seller=TimedFlowSeller(True)
            plan=seller.plan(observation(718),{'MILK':20,'WHEAT':15},{'WHEAT':14},
                             terminal=True,available_order_slots=slots)
            self.assertLessEqual(len(plan.orders),slots)
            for _,product,quantity in plan.orders:
                self.assertLessEqual(quantity,{'MILK':20,'WHEAT':1}[product])
            if slots>=2:
                self.assertEqual({p:q for _,p,q in plan.orders},{'MILK':20,'WHEAT':1})

    def test_ordered_sale_matches_engine(self):
        for product in engine.PRODUCTS:
            for position in (-1,0,1):
                farms=[engine._new_farm(10,3000) for _ in range(2)]
                private=[engine._new_private() for _ in range(2)]
                for s,n in enumerate((7,11)): private[s]['shed'][product]=n
                market=engine._new_market(); initial=market['inventory'][product]
                orders=[[['SELL',product,7]],[['SELL',product,11]]]
                if position<0: orders[1].insert(0,['SELL',product,0])
                if position>0: orders[0].insert(0,['SELL',product,0])
                states=[NS(observation=NS(farms=farms,private=private[s],market=market),action={'market':orders[s]}) for s in (0,1)]
                engine._process_market(states,NS(configuration={'maxMarketOrdersPerTurn':10,'boardSize':10,'shedCapacity':100,'farmHandCostMult':1}))
                expected=ordered_sale(product,7,11,initial,position)
                self.assertEqual(tuple(f['money']-3000 for f in farms),expected[:2])
                self.assertEqual(market['inventory'][product],expected[2])

    def test_entry_state_is_seat_local(self):
        entry=load_entry(); env=make('kaggriculture',configuration={'seed':0}); env.reset()
        for s in (0,1):
            action=entry.agent(*actor_input(env,s))
            self.assertLessEqual(len(action['market']),10)
        self.assertIsNot(entry._agents[0],entry._agents[1])
        self.assertIsNot(entry._agents[0].tapes,entry._agents[1].tapes)

if __name__=='__main__': unittest.main()
