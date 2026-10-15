"""Certify complete pasture lifetimes using unchanged official field rules."""
from collections import Counter,defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path

from common import AGENT
ASSET=AGENT/'asset.json.gz'
from market_game_20260930.ledger import engine

OUT=Path(__file__).parent
FIELD={'PLACE','FEED','CARE','HARVEST','COLLECT_FERTILIZER','DIG','BUILD_PASTURE'}


def simulate(slot,animal):
    farm=dict(tiles=[[None]*10 for _ in range(10)],farmer=[0,0],hands=[],unlocked_quadrants=[0])
    farm['tiles'][0][0]={'kind':'PASTURE'}
    private=dict(seeds={},shed={},inventories=[{animal:1,'WHEAT':1000}])
    schedule=defaultdict(list)
    for event in slot['events']:
        schedule[event['step']].append(event)
    dawns={}; harvests=[]; lost=[]; feeds=0; fertilizer=0
    product=engine.ANIMALS[animal]['product']
    for step in range(slot['place_step'],719):
        if step%24==0:
            dawns[str(step//24)]=copy.deepcopy(farm['tiles'][0][0])
        for event in schedule[step]:
            command=list(event['action'])
            if step==slot['place_step'] and command[0]=='PLACE':
                command[1]=animal
            bag=private['inventories'][0]
            before=dict(bag)
            engine._apply_unit_action(farm,private,0,command,10,step//24,24,100)
            feeds+=before.get('WHEAT',0)-bag.get('WHEAT',0)
            fertilizer+=bag.get('FERTILIZER',0)-before.get('FERTILIZER',0)
            if command[0]=='HARVEST':
                harvests.append(dict(step=step,unit=event['unit'],quantity=bag.get(product,0)-before.get(product,0)))
        if (step+1)%24==0:
            engine._daily_refresh_animals(farm,step//24)
        tile=farm['tiles'][0][0]
        if not isinstance(tile,dict) or tile.get('animal')!=animal:
            if not lost:
                lost.append(step)
    return dict(animal=animal,product=product,dawns=dawns,harvests=harvests,
                total=sum(r['quantity'] for r in harvests),feeds=feeds,fertilizer=fertilizer,escape_steps=lost)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/local/pasture_certificates.json'))
    args=parser.parse_args()
    target=args.out
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with gzip.open(ASSET,'rt',encoding='utf-8') as f:
        asset=json.load(f)
    loc=asset['routes'][0]['edges'][0]['example']; source=asset['sources'][loc]
    slots=[]; active={}; cargo={}; timeline={}
    for day in range(30):
        for hour,row in enumerate(source['tapes'][str(day)]):
            step=day*24+hour; timeline[step]=row
            for u,(action,xy) in enumerate(zip([row['farmer'],*row['hands']],row['positions'])):
                xy=tuple(xy)
                if action[0]=='PICKUP' and action[1] in ('COW','SHEEP'):
                    assert action[2]==1 and (u,action[1]) not in cargo
                    cargo[u,action[1]]=dict(step=step,unit=u,action=list(action))
                if action[0]=='PLACE' and action[1] in ('COW','SHEEP') and not row['unit_events'][u]['shed']:
                    assert xy not in active
                    pickup=cargo.pop((u,action[1]))
                    slot=dict(id=f'd{day}_{xy[0]}_{xy[1]}',xy=list(xy),animal=action[1],
                              place_step=step,pickup=pickup,events=[])
                    slots.append(slot); active[xy]=slot
                if xy in active and action[0] in FIELD:
                    active[xy]['events'].append(dict(step=step,unit=u,action=list(action)))
    assert not cargo
    for slot in slots:
        old=simulate(slot,slot['animal']); new=simulate(slot,'SHEEP' if slot['animal']=='COW' else 'COW')
        x,y=slot['xy']
        for day,tile in old['dawns'].items():
            assert source['dawns'][day]['tiles'][y][x]==tile,(slot['id'],day,'original dawn mismatch')
        dependencies=[]; deliveries=[]
        for old_h,new_h in zip(old['harvests'],new['harvests']):
            assert old_h['step']==new_h['step'] and old_h['unit']==new_h['unit']
            step=old_h['step']; day,hour=divmod(step,24); u=old_h['unit']
            delivery=(day+1)*24 if day<29 else None
            kind='night'; drop=None
            for t in range(step+1,min(720,(day+1)*24)):
                row=timeline[t]; commands=[row['farmer'],*row['hands']]
                if u>=len(commands):
                    continue
                c=commands[u]
                if c[0]=='DROP':
                    if row['unit_events'][u]['shed'].get(old['product'],0)<old_h['quantity']:
                        dependencies.append(dict(step=t,reason='partial_drop'))
                    delivery=t; kind='DROP'; drop=t
                    break
                if c[0] in ('PLACE','PICKUP') and len(c)>1 and c[1] in ('MILK','WOOL'):
                    dependencies.append(dict(step=t,reason='named_product_transfer',action=c))
                    if c[0]=='PLACE' and c[1]==old['product'] and c[2]>=old_h['quantity']:
                        delivery=t; kind='PLACE'
                        break
            if delivery is None:
                dependencies.append(dict(step=step,reason='terminal_no_delivery'))
            deliveries.append(dict(harvest_step=step,unit=u,step=delivery,kind=kind,drop=drop,
                                   original_quantity=old_h['quantity'],alternative_quantity=new_h['quantity']))
        pday,phour=divmod(slot['pickup']['step'],24)
        slot.update(original=old,alternative=new,deliveries=deliveries,dependencies=dependencies,
                    decision_day=pday-(phour<=1),eligible=pday>=3 and old['escape_steps']==new['escape_steps'] and not dependencies)
        assert old['feeds']==new['feeds'] and old['fertilizer']==new['fertilizer']
    totals=Counter()
    for slot in slots:
        totals[slot['original']['product']]+=slot['original']['total']
    assert dict(totals)=={'MILK':192,'WOOL':170},totals
    summary=dict(slots=len(slots),eligible=sum(s['eligible'] for s in slots),original_totals=dict(totals),
                 decisions={str(d):[s['id'] for s in slots if s['eligible'] and s['decision_day']==d] for d in sorted({s['decision_day'] for s in slots if s['eligible']})})
    result=dict(version='pasture_lifetime_certificate_v1',source_loc=loc,summary=summary,slots=slots,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ASSET,Path(engine.__file__),Path(__file__))})
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary)); print(json.dumps([dict(id=s['id'],old=s['original']['total'],new=s['alternative']['total'],eligible=s['eligible'],reasons=s['dependencies']) for s in slots]))


if __name__=='__main__':
    main()
