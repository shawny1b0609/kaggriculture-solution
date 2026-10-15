"""Portable, no-sklearn inference for exported small Poisson forests."""
import json
import math
from pathlib import Path
from functools import lru_cache

DEFAULT=Path(__file__).resolve().parent/'opponent_flow_v1.json'


class FlowModel:
    def __init__(self,path=DEFAULT):
        self.payload=json.loads(Path(path).read_text(encoding='utf-8'))
        self.heads=self.payload['heads']

    def predict_one(self,x):
        values=[]
        for head in self.heads:
            if 'constant' in head:
                values.append(head['constant']);continue
            total=head['base']
            for nodes in head['trees']:
                index=0
                while True:
                    feature,threshold,left,right,value,missing_left=nodes[index]
                    if feature<0:
                        total+=value;break
                    v=x[feature]
                    index=(left if missing_left else right) if math.isnan(v) else (left if v<=threshold else right)
            values.append(math.exp(min(15.,max(-30.,total))))
        return values


@lru_cache(maxsize=1)
def load_model():
    return FlowModel()
