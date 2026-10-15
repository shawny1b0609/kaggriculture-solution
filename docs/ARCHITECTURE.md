# Architecture and code map

The final agent combines a deterministic production schedule with constrained,
observation-dependent production choices and a learned market forecast. It
does not run the earlier node-selection RL experiments.

![Complete architecture](report/figures/detail/architecture_detailed.png)

## 1. Entry point and state

[`agent/main.py`](../agent/main.py) loads one immutable production asset and
creates a separate stateful actor for each seat. A new day-0/hour-0 observation
resets that seat. Only public observations and the player's own private
inventory enter decisions. Environment seeds and future shop events are not
passed to the actor by the evaluation wrappers.

The entry point composes `PastureAgent`, inherited crop and execution layers,
and `TimedFlowSeller(True)`. The trained market model is a JSON tree ensemble;
inference does not require scikit-learn.

## 2. Production skeleton and recovery

[`asset.json.gz`](../agent/asset.json.gz) contains the compiled source schedule,
daily reference farms, actions, unit-level product transfers and expected
production. Its source is episode 113905606, seat 0.

| Responsibility | Code under `agent/implementation/` |
|---|---|
| Asset loading and route infrastructure | `tools/portable_tree_agent.py` |
| Production cards and inherited route adapter | `compact_agent.py` |
| Hourly execution and protected market scheduling | `runtime/grid_agent.py` |
| Source action alignment | `runtime/source_tape.py` |
| Procurement and daily resource requirements | `runtime/resource_contract.py`, `runtime/stock.py` |
| Random weed recovery | `runtime/weed_repair.py` |
| Local deferred cash jobs | `runtime/cash_repair.py`, `tools/resilient_crop_debt_agent.py` |
| Stable continuation and day-boundary guards | `tools/stable_execution_agent.py`, `tools/slot_guard_experiment.py`, `tools/capacity_guard_experiment.py` |

An episode's `steps[t+1][seat].action` drives the transition from observation
`t` to observation `t+1`. Compiled tapes already resolve this delay; shifting
them again is an error. The game has 720 stored frames and 719 actions.

The executor reserves production inputs, orders them before use, maintains
worker identities and spawn positions, and repairs local disruptions. Repairs
can cost labor or production; they are not a proof that every possible game
state is recoverable. Runtime checks compare the actual selected plan, not
only the untouched source replay.

## 3. Compatible crop choices

[`g4/cycles.py`](../agent/market_game_20260930/g4/cycles.py) constructs
per-match copies of eligible wheat/carrot cycles.
[`g5/agent.py`](../agent/market_game_20260930/g5/agent.py) selects synchronized
choices using current demand, price impact and expected rival supply.

The deployed candidate pool contains 40 checked cycles across 15 days,
starting at zero-indexed day 10. Up to two compatible choices sharing a
delivery time may be selected together. A replacement changes seed commands,
production quantities, carried products, warehouse arrivals and reference
states, while preserving the movement schedule and a common return state.

[`scripts/certify_crops.py`](../scripts/certify_crops.py) reproduces single-tile
crop compatibility checks. Whole-farm delivery and resource compatibility are
enforced by the runtime adapter and full-game checks.

## 4. Compatible livestock choices

[`g8/pastures.py`](../agent/market_game_20260930/g8/pastures.py) recompiles a
cow/sheep placement throughout its lifetime. Their first yield dates,
intervals and care bonuses differ, so changing only the purchase item is
insufficient.

[`g8/agent.py`](../agent/market_game_20260930/g8/agent.py) values each candidate
against the already-committed milk/wool portfolio. The estimate includes
known shop consumption, rival visible capacity, short-term public-flow
forecasts, own price impact and animal purchase-cost differences. It blends
spot and projected values to reduce sensitivity to long extrapolations.

Eligibility covers 7 of 14 source animal placements. Decisions can occur on
days 3, 4, 9, 11 and 12. Cash below 2,000 or unresolved recovery blocks changes;
an estimated gain of at least 150 and a maximum of two changes per decision
day limit churn. The actual final evaluation exercised late sheep-to-cow
changes, not every possible certified substitution.

[`scripts/certify_livestock.py`](../scripts/certify_livestock.py) regenerates
the lifetime certificates from the production asset and official rules.

## 5. Market model

| Responsibility | Code under `agent/market_game_20260930/` |
|---|---|
| Reconcile own fills, separate public rival net flow | `ledger.py` |
| Online history and 16 public features | `g2/features.py` |
| Portable Poisson-tree predictor | `g2/model.py`, `g2/opponent_flow_v1.json` |
| Scenario search over sale quantities and timing | `g2/seller.py` |
| Inherited liquidity and competitive-market rules | `competitive_seller.py` |

Four regression heads predict rival sale volumes over `[0,1)`, `[1,4)`,
`[4,12)` and `[12,24)` turns. Nine scenarios combine three flow scales and
three relative sale-order positions. Candidate quantities include zero,
quarter, half, three-quarters and all available units, plus a baseline choice.
The objective balances expected own revenue, downside risk and holding cost.
The final deployed coefficient on directly penalizing opponent income is
zero; competition is represented through shared price impact and rival flow.

Liquidity protection keeps stock-age windows short below 10,000 cash.
Surplus goods can wait longer after that; the 24-turn search horizon is not
the same as the inventory age limit. Terminal liquidation and production-input
reservations remain active.

```python
farm_orders = plan_required_purchases_and_hires(observation)
sale_slots = max(0, 10 - len(farm_orders))
sale_orders = seller.plan(
    sellable_inventory, protected_inputs, available_order_slots=sale_slots
)
# Optional sales may settle first, but cannot evict a required farm order.
orders = sale_orders + farm_orders
```

## Naming

The historical source paths use experiment names. `g2` / `LEARN24` is the
forecast-guided seller, `g5` / `SYNC2` is synchronized crop adaptation, and
`g8` / `PASTURE` adds adaptive livestock. `731291` identifies the inherited
production root. Files under `node_rl` are imported infrastructure; their
presence does not mean the final agent uses an RL checkpoint.
