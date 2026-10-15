"""SYNC2 crops, compatible public pasture allocation, frozen G2 market."""
from pathlib import Path
import sys

_tree = None
_cards = None
_agents = {}
_farm_class = None
_seller_class = None


def agent(observation, configuration=None):
    global _tree, _cards, _farm_class, _seller_class
    if _tree is None:
        raw = configuration.get('__raw_path__') if hasattr(configuration, 'get') else None
        package = Path(str(raw)).resolve().parent if raw else Path(globals().get('__file__', 'main.py')).resolve().parent
        sys.path[:0] = [str(package/'implementation'), str(package)]
        from tools.portable_tree_agent import PortableTree
        from compact_agent import make_cards
        from market_game_20260930.g8.agent import PastureAgent
        from market_game_20260930.g2.seller import TimedFlowSeller
        _tree = PortableTree(package/'asset.json.gz')
        _cards = make_cards(_tree)
        _farm_class, _seller_class = PastureAgent, TimedFlowSeller
    seat = int(observation['player'])
    if seat not in _agents or (int(observation['day']) == 0 and int(observation['hour']) == 0):
        actor = _farm_class(mode='public', tree=_tree, cards=_cards)
        actor.market_seller = _seller_class(True)
        _agents[seat] = actor
    actor = _agents[seat]
    action = actor(observation, configuration)
    actor.market_seller.record_action(observation, action, configuration)
    return action
