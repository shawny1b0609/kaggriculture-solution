"""Frozen identities for the featured-program forest. Do not wrap 1911."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

PLAN_ID = "F20260925.1_featured_program_forest"
SCOPE_ID = "featured_frozen_20260922"
CUTOFF_DATE = "2026-09-22"
ENGINE_VERSION = "1.32.7"
ENGINE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
STATES = 720
ACTIONS = 719

P1911 = {
    "submission_id": "56199579",
    "name": "rescue_pasture",
    "historical_score_snapshot": 1911.5,
    "archive": ROOT / "artifacts/online_iteration_20260912/final_two_rescue_20260913_v1/submission_rescue_pasture_20260913_6fc03f9f1b9a.tar.gz",
    "archive_sha256": "8aa96af8c35bc298e48346e35407b892dc7c03ac3c9abf280fa080758b75b785",
    "main_sha256": "6fc03f9f1b9ac08c6dbabeb9023388b051c9a5e1dbd48fdfd8d6bebf9f3cbc9a",
}

FEATURED = ROOT / "data" / "featured"
FEATURED_RECENT = ROOT / "data" / "featured_recent"
FEATURED_INDEX = ROOT / "data" / "featured_index" / "refresh_20260924_T2a" / "manifest.csv"

ARTIFACTS = HERE / "artifacts"
EVENTS = ARTIFACTS / "events"
PROGRAMS = ARTIFACTS / "programs"
GRAPHS = ARTIFACTS / "graphs"
REPORTS = ARTIFACTS / "reports"
ATTEMPTS = ARTIFACTS / "attempts" / "attempts.jsonl"
STATE_PATH = HERE / "state.json"

ATTEMPT_CAP = 6000
PHASE_CAPS = {
    "F0": 64,
    "F1_data_parse": 0,
    "F2": 800,
    "F3": 960,
    "F4": 400,
    "F5": 512,
    "F6": 3072,
    "reserve": 192,
}

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
SELLABLE = ("MILK", "WOOL", "STRAWBERRY", "MELON", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER")
FEED = "WHEAT"
SHOP_NAMES = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
MODELS = ARTIFACTS / "models"
FOREST_NODES = GRAPHS / "path_forest_close_nodes.jsonl"
FOREST_ROOTS = GRAPHS / "roots_close.json"
FOREST_WEIGHTS = PROGRAMS / "forest_root_weights.json"
TRAIN_SEED_START = 8826502701
RL_SEED_START = 8826502901
DUAL_SEED_START = 8826503101
DUAL_CONTINUE_SEED = 8826503613
CLOSE_REL = 0.15
CLOSE_FLOOR = 20000.0
GRID_NODES = GRAPHS / "grid_net_switch_nodes.jsonl"
GRID_EDGES = GRAPHS / "grid_net_switch_edges.jsonl"
GRID_WEIGHTS = PROGRAMS / "grid_root_weights.json"
GRID_DUAL_SEED_START = 8826504101
GRID_MAX_CHOICES = 384

# New seeds for this branch. Do not reuse reconstruction holdouts (2026092501+).
F0_SEED = 8826502501
EVAL_SEED_START = 8826502601
