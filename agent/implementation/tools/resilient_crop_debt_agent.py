"""Execution v4: close a tracked crop debt at its source replant boundary."""
import copy
from tools.resilient_contract_agent import ResilientContractMixin, delay_worker_chain
from runtime.grid_agent import own_farm, engine
from runtime.weed_repair import _units, _replace_idle_action, MOVES
from runtime.cash_repair import _plant_recovered_before_midnight


def clear_stale_crop_before_replant(obs, tape, hour, debt_coordinates, protected=()):
    """Repair only known debt, never clear arbitrary healthy source crops.

    Field simulation includes earlier workers this hour. If their HARVEST or
    DIG already clears the tile, leave it alone. The insertion is committed
    only with a same-day PLANT/WATER suffix and safe cross-worker ordering.
    """
    if hour >= len(tape) or not debt_coordinates:
        return [], []
    farm,private=copy.deepcopy(own_farm(obs)),copy.deepcopy(obs.get('private') or {})
    positions=[farm['farmer'],*farm.get('hands',[])]
    fixed,blocked=[],[]
    for worker,action in enumerate(_units(tape[hour])):
        if worker>=len(positions):
            continue
        xy=tuple(positions[worker])
        expected=tape[hour].get('positions') or []
        tile=farm['tiles'][xy[1]][xy[0]]
        if (action == ['HARVEST'] and xy in debt_coordinates
                and worker<len(expected) and list(xy)==expected[worker]
                and isinstance(tile,dict) and tile.get('kind')=='PLANT'
                and int(obs['day'])-int(tile['planted_day'])
                    < int(engine.CROPS[tile['crop']]['first_yield_day'])):
            # A late crop has missed its source harvest deadline. HARVEST is
            # provably a no-op today. If the next tile job is replanting, use
            # that already allocated hour to clear it rather than shift the
            # farmer (whose final position affects tomorrow's births).
            next_job = None
            for h in range(hour,len(tape)):
                for w,op in enumerate(_units(tape[h])):
                    pos=tape[h].get('positions') or []
                    if ((h,w) <= (hour,worker) or w>=len(pos)
                            or tuple(pos[w])!=xy or op[0] in MOVES or op==['PASS']):
                        continue
                    next_job=op
                    break
                if next_job is not None:
                    break
            if next_job and next_job[0]=='PLANT':
                _replace_idle_action(tape[hour],worker,['DIG'])
                action=['DIG']
                fixed.append({'hour':hour,'worker':worker,'xy':list(xy),
                    'crop':next_job[1],'removed_crop':tile['crop'],
                    'removed_yield':tile.get('yield_units',0),
                    'mode':'replace_ineffective_harvest_before_replant'})
        if (len(action)>1 and action[0]=='PLANT' and xy in debt_coordinates
                and worker<len(expected) and list(xy)==expected[worker]
                and isinstance(tile,dict) and tile.get('kind')=='PLANT'):
            trial=copy.deepcopy(tape)
            detail=delay_worker_chain(trial,worker,hour,['DIG'],protected)
            row={'hour':hour,'worker':worker,'xy':list(xy),'crop':action[1],
                 'removed_crop':tile.get('crop'),'removed_yield':tile.get('yield_units',0)}
            if detail and _plant_recovered_before_midnight(trial,hour,xy,action[1]):
                tape[:]=trial
                fixed.append({**row,**detail})
                action=['DIG']
            else:
                blocked.append(row)
        engine._apply_unit_action(farm,private,worker,action,len(farm['tiles']),int(obs['day']),24,100)
    return fixed,blocked


class ResilientCropDebtMixin(ResilientContractMixin):
    def _play(self,obs,hour):
        coordinates={tuple(row['xy']) for row in self.deferred_weed_jobs+self.deferred_cash_jobs}
        fixed,blocked=clear_stale_crop_before_replant(obs,self.tape,hour,coordinates,self.weed_plan)
        for row in fixed:
            self._choice_log('execution_v2',{'day':obs['day'],'kind':'close_stale_crop_debt',**row})
        for row in blocked:
            self._choice_log('execution_v2_unresolved',{'day':obs['day'],'kind':'close_stale_crop_debt',**row})
        return super()._play(obs,hour)
