"""Per-match animal option compilation; all original resources stay read-only."""
from functools import lru_cache
import copy
import json
from pathlib import Path

from market_game_20260930.g4.cycles import CycleTapes,add_amount

CERTIFICATES=Path(__file__).with_name('pasture_certificates_v2.json')


@lru_cache(maxsize=1)
def certificates():
    return json.loads(CERTIFICATES.read_text(encoding='utf-8'))


class PastureTapes(CycleTapes):
    def __init__(self,original,loc):
        super().__init__(original,loc)
        self.selected_animals=[]

    def choose_animal(self,slot):
        if not slot['eligible'] or slot['dependencies']:
            raise ValueError('uncertified animal option')
        if slot['id'] in self.selected_animals:
            raise ValueError('animal selected twice')
        old,new=slot['original'],slot['alternative']
        assert old['escape_steps']==new['escape_steps']
        assert old['feeds']==new['feeds'] and old['fertilizer']==new['fertilizer']
        x,y=slot['xy']
        for day,tile in old['dawns'].items():
            assert self.original.dawn_farm(self.loc,int(day))['tiles'][y][x]==tile
        changes={}
        def at(step):
            day,hour=divmod(step,24)
            if day not in changes:
                changes[day]=self.day_tape(self.loc,day)
            return changes[day][hour]
        pickup=slot['pickup']; row=at(pickup['step']); unit=pickup['unit']
        command=[row['farmer'],*row['hands']][unit]
        assert command==['PICKUP',old['animal'],1]
        command[1]=new['animal']
        event=row['unit_events'][unit]['shed']
        assert event.get(old['animal'])==-1
        add_amount(event,old['animal'],1); add_amount(event,new['animal'],-1)
        add_amount(row['shed_delta_units'],old['animal'],1); add_amount(row['shed_delta_units'],new['animal'],-1)
        row=at(slot['place_step'])
        command=[row['farmer'],*row['hands']][unit]
        assert command==['PLACE',old['animal']] and row['positions'][unit]==slot['xy']
        command[1]=new['animal']
        for arrival in slot['deliveries']:
            row=at(arrival['harvest_step']); unit=arrival['unit']
            assert [row['farmer'],*row['hands']][unit]==['HARVEST']
            assert row['positions'][unit]==slot['xy']
            a,b=arrival['original_quantity'],arrival['alternative_quantity']
            assert row['yielded'].get(old['product'],0)>=a
            add_amount(row['yielded'],old['product'],-a); add_amount(row['yielded'],new['product'],b)
            if arrival['kind']=='DROP':
                row=at(arrival['drop'])
                assert [row['farmer'],*row['hands']][unit]==['DROP']
                event=row['unit_events'][unit]['shed']
                assert event.get(old['product'],0)>=a
                add_amount(event,old['product'],-a); add_amount(event,new['product'],b)
                add_amount(row['shed_delta_units'],old['product'],-a); add_amount(row['shed_delta_units'],new['product'],b)
            else:
                assert arrival['kind']=='night' and arrival['step']<=696
        # No partial mutation is visible if any certificate check failed.
        for day,tape in changes.items():
            self.overrides[self.loc,day]=tape
        for day,tile in new['dawns'].items():
            self.dawn_overrides[int(day)][(x,y)]=copy.deepcopy(tile)
        self.selected_animals.append(slot['id'])
