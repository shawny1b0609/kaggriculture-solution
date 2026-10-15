"""Shared portable paths and legal agent inputs."""
from pathlib import Path
import copy
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'agent'
sys.path[:0] = [str(AGENT / 'implementation'), str(AGENT), str(ROOT)]
CONFIG_KEYS = ('boardSize', 'episodeSteps', 'farmHandCostMult', 'marketParams',
               'maxMarketOrdersPerTurn', 'shedCapacity', 'startingMoney',
               'townCenterSellInterval', 'townShopSellInterval',
               'townShopUnlockInterval', 'turnsPerDay', 'weedSpawnChance')

def load_entry(path=AGENT / 'main.py'):
    spec = importlib.util.spec_from_file_location('submission_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def actor_input(env, seat):
    obs = copy.deepcopy(json.loads(json.dumps(env.state[seat].observation)))
    obs['step'] = int(obs['day']) * 24 + int(obs['hour'])
    cfg = {k:copy.deepcopy(env.configuration[k]) for k in CONFIG_KEYS if k in env.configuration}
    assert 'seed' not in obs and 'seed' not in cfg
    return obs, cfg

def assert_engine():
    import kaggle_environments
    if kaggle_environments.__version__ != '1.32.7':
        raise RuntimeError('This release requires kaggle-environments==1.32.7')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
