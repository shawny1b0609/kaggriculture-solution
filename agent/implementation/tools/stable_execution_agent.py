"""Public opt-in entrypoint for the accepted execution repair candidate.

Compose StableExecutionMixin BEFORE the existing execution agent. Keep the
existing seller, guards, graph, tape compiler and node selector unchanged.
This module does not monkeypatch the frozen runtime or restart any pipeline.
The experimental partial-procurement/seed-suffix variant is NOT included.
"""
from tools.resilient_crop_debt_agent import ResilientCropDebtMixin

EXECUTION_RULESET = 'contract_chain_capacity_crop_debt_v4_original_seller'
StableExecutionMixin = ResilientCropDebtMixin
