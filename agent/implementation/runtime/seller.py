"""Market sellers for the featured production forest.

``RollingLiquiditySeller`` separates two concerns which used to be mixed:

* production reservations and hard operating cash are constraints;
* the remaining inventory is timed against the public price curve.

The class never buys anything and never consumes reserved stock.  It may force
an early sale when a same-turn purchase/hire or shed-capacity obligation would
otherwise fail.  Outside those cases it uses a cheap rolling two-stage value
calculation, public town demand and observed opponent selling pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math

from kaggle_environments.envs.kaggriculture import kaggriculture as engine
from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS, TOWN_CENTER_PRODUCTS, market_price

from config import SELLABLE
from runtime.view import own_farm, rival_farm, step_of

RESERVE_WHEAT_PER_ANIMAL = 2


def shop_demand(shops, item, step) -> int:
    amount = 0
    if step % 4 == 0:
        for shop in shops:
            products = SHOPS.get(shop) or []
            if item in products:
                amount += 2 if len(products) == 1 else 1
    if step % 24 == 0 and item in TOWN_CENTER_PRODUCTS:
        amount += 1
    return amount


def quote(item, inventory, qty, params=None) -> float:
    total = 0.0
    inv = int(inventory)
    for _ in range(max(0, int(qty))):
        price = market_price(item, inv, params)
        total += price
        if price > 1:
            inv += 1
    return total


def sell(obs, available: dict[str, int], reserved: dict[str, int]) -> list:
    """Sell non-reserved shed goods. Immediate if price>1; keep wheat reserve."""
    step = step_of(obs)
    market = obs.get("market") or {}
    inventory = market.get("inventory") or {}
    params = market.get("params")
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    orders = []
    for item in SELLABLE:
        have = int(available.get(item) or 0) - int(reserved.get(item) or 0)
        if have <= 0:
            continue
        inv = int(inventory.get(item) or 0)
        now = quote(item, inv, have, params)
        wait = quote(item, inv - shop_demand(shops, item, step + 1), have, params)
        # Short window: sell now unless waiting one shop tick is clearly better and not last day.
        day = int(obs.get("day") or 0)
        if day >= 28 or now >= wait * 0.98:
            orders.append(["SELL", item, have])
        elif have > 4:
            orders.append(["SELL", item, max(1, have // 2)])
    return orders[:8]


def sell_immediate(available: dict[str, int], reserved: dict[str, int]) -> list[list]:
    """Liquidate every sellable shed surplus at the next market phase.

    No price timing: only stock committed to the selected production tape is
    retained. The caller may defer orders if the ten-order turn limit is full.
    """
    return [
        ["SELL", item, int(available.get(item) or 0) - int(reserved.get(item) or 0)]
        for item in SELLABLE
        if int(available.get(item) or 0) > int(reserved.get(item) or 0)
    ]


def sale_revenue(item: str, quantity: int, inventory: int, params=None) -> tuple[float, int]:
    """Exact own-side revenue on the official per-unit monotone price curve."""
    revenue = 0.0
    cursor = int(inventory)
    for _ in range(max(0, int(quantity))):
        price = float(market_price(item, cursor, params))
        revenue += price
        if price > 1:
            cursor += 1
    return revenue, cursor


def lockstep_sale_revenue(item: str, quantity: int, rival_quantity: int,
                          inventory: int, params=None) -> tuple[float, int]:
    """Exact own revenue when the rival sells the same item in the same slot.

    The official engine quotes both players from the same pre-commit inventory,
    then commits both units.  Thus a mirrored pair of sales advances public
    inventory by two between our consecutive units, rather than by one.  Any
    rival tail continues to affect the inventory seen by a later sale.
    """
    own_left = max(0, int(quantity))
    rival_left = max(0, int(rival_quantity))
    cursor = int(inventory)
    revenue = 0.0
    while own_left > 0:
        price = float(market_price(item, cursor, params))
        revenue += price
        if price > 1:
            cursor += 1
        own_left -= 1
        if rival_left > 0:
            # The rival received the same quote before either unit committed.
            if price > 1:
                cursor += 1
            rival_left -= 1
    while rival_left > 0:
        price = float(market_price(item, cursor, params))
        if price > 1:
            cursor += 1
        rival_left -= 1
    return revenue, cursor


@dataclass(slots=True)
class SellPlan:
    orders: list[list] = field(default_factory=list)
    expected_revenue: float = 0.0
    forced: bool = False
    funding_target: float = 0.0
    funding_gap: float = 0.0
    funding_shortfall: float = 0.0
    capacity_release: int = 0
    timing_quantities: dict[str, int] = field(default_factory=dict)
    forced_quantities: dict[str, int] = field(default_factory=dict)
    timing_details: dict[str, dict] = field(default_factory=dict)


class RollingLiquiditySeller:
    """Fast rolling seller with hard liquidity and capacity guarantees.

    The price-timing layer solves a two-stage liquidation problem for each
    unreserved product.  The future stage is evaluated at several public-demand
    horizons.  Our own price impact is exact; opponent supply is an observable
    EWMA/visible-farm forecast.  A separate forced layer then adds the smallest
    high-value set of units needed for operating cash or shed room.

    This is intentionally formula based.  Route validation can therefore blame
    a remaining failure on a concrete cash impossibility, order congestion or
    field execution instead of on an opaque learned policy.
    """

    def __init__(self, horizon_turns: int = 72, price_window_turns: int = 4,
                 low_cash_wait_penalty: float = 0.002) -> None:
        self.horizon_turns = max(4, int(horizon_turns))
        self.price_window_turns = max(4, min(48, int(price_window_turns)))
        self.bootstrap_window_turns: int | None = None
        self.low_cash_takeprofit_ratio: float | None = None
        self.low_cash_wait_penalty = max(0.002, float(low_cash_wait_penalty))
        self.last_inventory: dict[str, int] | None = None
        self.last_step: int | None = None
        self.own_last_sales: dict[str, int] = {}
        self.opponent_pressure: dict[str, float] = {}
        self.timing_details: dict[str, dict] = {}
        # Opt-in for the repaired seller; legacy checkpoints keep their model.
        self.account_for_arrivals = False

    @staticmethod
    def _town_demand(obs: dict, start: int, turns: int) -> dict[str, int]:
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        result = {item: 0 for item in SELLABLE}
        for offset in range(max(0, int(turns))):
            future = int(start) + offset
            if future % 4 == 0:
                for item in SELLABLE:
                    result[item] += shop_demand(shops, item, future)
            elif future % 24 == 0:
                # ``shop_demand`` includes town-center demand only when called
                # on a matching step.  The shop branch above already includes
                # it on steps divisible by both 4 and 24.
                for item in TOWN_CENTER_PRODUCTS:
                    if item in result:
                        result[item] += 1
        return result

    def update(self, obs: dict) -> None:
        """Infer the rival's recent market additions from public inventory."""
        step = step_of(obs)
        if self.last_step == step:
            # Branch selection and order planning can inspect the same public
            # observation. A second read must not erase the pending own sale.
            return
        current = {
            item: int(((obs.get("market") or {}).get("inventory") or {}).get(item) or 0)
            for item in SELLABLE
        }
        if self.last_inventory is not None and self.last_step is not None and step == self.last_step + 1:
            consumed = self._town_demand(obs, self.last_step, 1)
            for item in SELLABLE:
                residual = (
                    current[item]
                    - int(self.last_inventory.get(item) or 0)
                    - int(self.own_last_sales.get(item) or 0)
                    + int(consumed.get(item) or 0)
                )
                old = float(self.opponent_pressure.get(item) or 0.0)
                self.opponent_pressure[item] = 0.75 * old + 0.25 * max(0.0, float(residual))
        elif self.last_step is not None and step <= self.last_step:
            self.opponent_pressure = {}
        self.last_inventory = current
        self.last_step = step
        self.own_last_sales = {}

    def stressed_revenue(self, obs: dict, orders: list[list]) -> float:
        """Conservative cash estimate for the selected same-slot sale orders."""
        market = obs.get("market") or {}
        inventory = market.get("inventory") or {}
        params = market.get("params")
        total = 0.0
        for order in orders:
            if not order or order[0] != "SELL" or len(order) < 3:
                continue
            item, quantity = str(order[1]), max(0, int(order[2] or 0))
            pressure = max(0, int(math.ceil(float(self.opponent_pressure.get(item) or 0.0))))
            revenue, _ = lockstep_sale_revenue(
                item, quantity, min(quantity, pressure), int(inventory.get(item) or 0), params,
            )
            total += revenue
        return total

    @staticmethod
    def _visible_supply(farm: dict, day: int, horizon_days: int) -> dict[str, float]:
        supply = {item: 0.0 for item in SELLABLE}
        for row in farm.get("tiles") or []:
            for tile in row or []:
                if not isinstance(tile, dict):
                    continue
                held = max(0, int(tile.get("yield_units") or 0))
                if tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop") or "")
                    rule = engine.CROPS.get(crop)
                    if not rule or crop not in supply:
                        continue
                    supply[crop] += held
                    age = int(day) - int(tile.get("planted_day") or day)
                    if rule.get("ongoing"):
                        first = int(rule["first_yield_day"])
                        last = int(rule["max_yield_day"])
                        interval = max(1, int(rule["interval"]))
                        for offset in range(1, horizon_days + 1):
                            future_age = age + offset
                            if first <= future_age and (future_age - first) % interval == 0:
                                supply[crop] += 1
                    elif age < int(rule["max_yield_day"]) <= age + horizon_days:
                        supply[crop] += max(1, int(rule["max_yield"]) - held)
                animal = str(tile.get("animal") or "")
                rule = engine.ANIMALS.get(animal)
                if not rule:
                    continue
                product = str(rule["product"])
                if product not in supply:
                    continue
                supply[product] += held
                age = int(day) - int(tile.get("placed_day") or day)
                first = int(rule["first_yield_day"])
                interval = max(1, int(rule["interval"]))
                for offset in range(1, horizon_days + 1):
                    future_age = age + offset
                    if future_age >= first and (future_age - first) % interval == 0:
                        supply[product] += 1
        return supply

    def _forecast_delta(self, obs: dict, item: str, turns: int,
                        visible_supply: dict[str, float]) -> int:
        demand = self._town_demand(obs, step_of(obs), turns).get(item, 0)
        horizon_days = max(1, math.ceil(self.horizon_turns / 24))
        visible_fraction = min(1.0, float(turns) / max(1.0, horizon_days * 24.0))
        # ``opponent_pressure`` is an EWMA, not a permanent per-turn flow.
        # Project its decaying tail; multiplying it by the whole horizon turns
        # one observed sale into a fictitious multi-day liquidation panic.
        recent = float(self.opponent_pressure.get(item) or 0.0)
        recent_tail = recent * (1.0 - 0.75 ** max(0, int(turns))) / (1.0 - 0.75)
        rival = (
            recent_tail
            + float(visible_supply.get(item) or 0.0) * visible_fraction
        )
        return int(round(rival - float(demand)))

    @staticmethod
    def _revenue_prefix(item: str, inventory: int, quantity: int, params=None) -> list[float]:
        prefix = [0.0]
        cursor = int(inventory)
        for _ in range(max(0, int(quantity))):
            price = float(market_price(item, cursor, params))
            prefix.append(prefix[-1] + price)
            if price > 1:
                cursor += 1
        return prefix

    def _timed_quantity(self, obs: dict, item: str, stock: int,
                        projected_arrival: int, visible_supply: dict[str, float]) -> int:
        """Risk-adjusted visible-information MPC for current自产 inventory.

        Candidate quantities follow the sell-policy grid: hold, 25%, 50%,
        75%, or all.  They share one near, publicly known town-demand window.
        Future bagged produce is not counted as legal shed inventory or cash.
        """
        if stock <= 0:
            return 0
        step = step_of(obs)
        cash = float(own_farm(obs).get("money") or 0.0)
        window = (max(4, int(self.bootstrap_window_turns))
                  if cash < 10000 and self.bootstrap_window_turns is not None
                  else self.price_window_turns)
        remaining = max(0, 718 - step)
        if remaining <= 0:
            self.timing_details[item] = {
                "reason": "TERMINAL_LIQUIDATION", "quantity": stock,
                "evaluation_mode": "visible_scenario_mpc",
            }
            return stock
        params = (obs.get("market") or {}).get("params")
        inventory = int(((obs.get("market") or {}).get("inventory") or {}).get(item) or 0)
        base = float((params or engine.MARKET_PARAMS)[item]["base"])
        if cash < 10000 and self.low_cash_takeprofit_ratio is not None:
            full_revenue, _ = sale_revenue(item, stock, inventory, params)
            average_quote = full_revenue / stock
            if average_quote >= base * self.low_cash_takeprofit_ratio:
                self.timing_details[item] = {
                    "reason": "BOOTSTRAP_TAKE_PROFIT", "quantity": stock,
                    "average_quote": round(average_quote, 3),
                    "base_price": round(base, 3),
                    "evaluation_mode": "low_cash_premium_quote",
                }
                return stock
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        demand_ticks = [
            tick for tick in range(step, min(718, step + window))
            if shop_demand(shops, item, tick) > 0 and tick + 1 <= 718
        ]
        if not demand_ticks:
            self.timing_details[item] = {
                "reason": "NO_NEAR_KNOWN_DEMAND", "quantity": stock,
                "evaluation_mode": "known_demand_window",
            }
            return stock
        # Only publicly known shops enter the forecast.  The final horizon
        # includes all currently visible demand within the bounded window;
        # unknown future unlocks are never leaked into this calculation.
        horizons = sorted({tick + 1 - step for tick in demand_ticks})
        candidates = sorted({
            0,
            stock,
            *[min(stock, max(1, int(math.ceil(stock * fraction))))
              for fraction in (0.25, 0.50, 0.75)],
        })
        weights = (0.20, 0.50, 0.30)
        risk_weight = 0.40
        pressure = max(0.0, float(self.opponent_pressure.get(item) or 0.0))
        working_capital_gap = max(0.0, min(1.0, (10000.0 - cash) / 10000.0))
        wait_penalty = 0.002 + working_capital_gap * (self.low_cash_wait_penalty - 0.002)
        horizon_days = max(1, math.ceil(self.horizon_turns / 24))
        best_score = -float("inf")
        best_now = 0
        best_detail: dict = {}
        for quantity in candidates:
            candidate_best = -float("inf")
            candidate_detail = None
            for turns in horizons:
                demand = int(self._town_demand(obs, step, turns).get(item) or 0)
                fraction = min(1.0, float(turns) / max(1.0, horizon_days * 24.0))
                visible = max(0.0, float(visible_supply.get(item) or 0.0)) * fraction
                recent = pressure * (1.0 - 0.75 ** max(0, int(turns))) / (1.0 - 0.75)
                rival_scenarios = (
                    0.25 * recent,
                    0.75 * recent + 0.25 * visible,
                    1.25 * recent + 0.75 * visible,
                )
                # The caller supplies this day's known remaining production,
                # not future market information. Spread it over the remaining
                # day for valuation ONLY; never sell it or fund a bill with it.
                arrivals = (int(max(0, projected_arrival) * min(1., turns / max(1, 24 - int(obs.get('hour') or 0))))
                            if self.account_for_arrivals else 0)
                future_quantity = stock - quantity + arrivals
                values = []
                concurrent_units = []
                for scenario_index, rival in enumerate(rival_scenarios):
                    if scenario_index == 0:
                        rival_now = 0
                        rival_future = 0
                    elif scenario_index == 1:
                        expected_burst = max(0, int(math.ceil(pressure)))
                        rival_now = min(quantity, expected_burst)
                        rival_future = min(future_quantity, expected_burst)
                    else:
                        # The concentrated scenario explicitly covers a mirror
                        # or another opponent choosing the same liquidation.
                        rival_now = quantity
                        rival_future = future_quantity
                    now_revenue, after_now = lockstep_sale_revenue(
                        item, quantity, rival_now, inventory, params,
                    )
                    background_rival = max(0, int(round(rival)) - rival_now)
                    future_inventory = max(
                        0,
                        after_now + background_rival - demand,
                    )
                    future_revenue, _ = lockstep_sale_revenue(
                        item, future_quantity, rival_future, future_inventory, params,
                    )
                    values.append(now_revenue + future_revenue)
                    concurrent_units.append([rival_now, rival_future])
                mean = sum(weight * value for weight, value in zip(weights, values))
                downside = math.sqrt(sum(
                    weight * max(0.0, mean - value) ** 2
                    for weight, value in zip(weights, values)
                ))
                score = mean - risk_weight * downside - wait_penalty * turns * stock * (
                    1.0 - quantity / stock
                ) * float((params or engine.MARKET_PARAMS)[item]["base"])
                if score > candidate_best + 1e-9:
                    candidate_best = score
                    candidate_detail = {
                        "horizon": turns,
                        "demand": demand,
                        "estimated_own_arrivals": arrivals,
                        "concurrent_rival_units": concurrent_units,
                        "scenario_values": [round(value, 3) for value in values],
                        "mean": round(mean, 3),
                        "downside": round(downside, 3),
                        "score": round(score, 3),
                    }
            if (candidate_best > best_score + 1e-9
                    or (abs(candidate_best - best_score) <= 1e-9 and quantity > best_now)):
                best_score = candidate_best
                best_now = quantity
                best_detail = candidate_detail or {}

        current_price = float(((obs.get("market") or {}).get("prices") or {}).get(item)
                              or market_price(item, inventory, params))
        if best_now <= 0:
            reason = "HOLD_FOR_DEMAND"
        elif pressure >= 1.0:
            reason = "OPPONENT_SUPPLY_RISK"
        elif current_price >= base:
            reason = "FAVORABLE_CURRENT_PRICE"
        else:
            reason = "ROLLING_VALUE"
        self.timing_details[item] = {
            "reason": reason,
            "quantity": int(best_now),
            "stock": int(stock),
            "current_price": round(current_price, 3),
            "base_price": round(base, 3),
            "opponent_pressure": round(pressure, 3),
            "evaluation_mode": "known_demand_window",
            **best_detail,
        }
        return min(stock, max(0, int(best_now)))

    def _subset_for_obligation(
        self, free: dict[str, int], inventories: dict[str, int], params,
        funding_gap: float, release_units: int, slots: int,
    ) -> dict[str, int]:
        """Fund the obligation using stressed, per-unit market receipts."""
        items = [item for item in SELLABLE if int(free.get(item) or 0) > 0]
        slots = max(0, min(int(slots), len(items)))
        if slots <= 0 or (funding_gap <= 1e-9 and release_units <= 0):
            return {}

        best = None
        best_quantities: dict[str, int] = {}
        # A recent rival sale can coincide with ours at the same order index.
        # Quote both players from the same pre-commit inventory, as the engine
        # does.  Unknown future rival orders are not treated as certain cash.
        stressed_prefix = {}
        for item in items:
            stock = int(free[item])
            pressure = max(0, int(math.ceil(float(self.opponent_pressure.get(item) or 0.0))))
            rival_units = min(stock, pressure)
            cursor = int(inventories[item])
            prefix = [0.0]
            for unit in range(stock):
                price = float(market_price(item, cursor, params))
                prefix.append(prefix[-1] + price)
                if price > 1:
                    cursor += 1 + int(unit < rival_units)
            stressed_prefix[item] = prefix
        for width in range(1, slots + 1):
            for subset in combinations(items, width):
                marginal = []
                for item in subset:
                    for unit in range(int(free[item])):
                        prefix = stressed_prefix[item]
                        marginal.append((prefix[unit + 1] - prefix[unit], item))
                marginal.sort(reverse=True)
                quantities = {item: 0 for item in subset}
                revenue = 0.0
                units = 0
                for _price, item in marginal:
                    if revenue >= funding_gap - 1e-9 and units >= release_units:
                        break
                    # Each item's order sells a prefix, not arbitrary units.
                    # Reprice the actual prefix after selecting one more unit.
                    before = stressed_prefix[item][quantities[item]]
                    quantities[item] += 1
                    revenue += stressed_prefix[item][quantities[item]] - before
                    units += 1
                feasible = revenue >= funding_gap - 1e-9 and units >= release_units
                # Reliability first: feasible, then minimum inventory consumed,
                # then minimum order slots, then maximum immediate proceeds.
                score = (1 if feasible else 0, -units if feasible else revenue,
                         -width if feasible else units, revenue)
                if best is None or score > best:
                    best = score
                    best_quantities = quantities
            if best is not None and best[0] == 1:
                break
        return {item: quantity for item, quantity in best_quantities.items() if quantity > 0}

    def plan(
        self,
        obs: dict,
        available: dict[str, int],
        reserved: dict[str, int],
        *,
        minimum_cash: float = 0.0,
        capital_floor: float = 0.0,
        required_capacity: int = 0,
        available_order_slots: int = 10,
        projected_arrivals: dict[str, int] | None = None,
        terminal: bool = False,
    ) -> SellPlan:
        self.update(obs)
        cash = float(own_farm(obs).get("money") or 0)
        operating_target = max(0.0, float(minimum_cash))
        # During capital formation, turn saleable surplus into working cash
        # even when the next bill is already covered.  Only the smallest
        # stressed-value bundle needed for the floor is forced; timing sales
        # above that bundle still follow the visible-demand price model.
        funding_target = max(operating_target, max(0.0, float(capital_floor)))
        funding_gap = max(0.0, funding_target - cash)
        free = {
            item: max(0, int(available.get(item) or 0) - int(reserved.get(item) or 0))
            for item in SELLABLE
        }
        occupied = sum(max(0, int(value or 0)) for value in available.values())
        capacity_release = max(0, occupied + max(0, int(required_capacity)) - 100)
        slots = max(0, min(10, int(available_order_slots)))
        market = obs.get("market") or {}
        inventories = {item: int((market.get("inventory") or {}).get(item) or 0) for item in SELLABLE}
        params = market.get("params")
        projected_arrivals = projected_arrivals or {}
        self.timing_details = {}

        if terminal:
            quantities = dict(free)
            timing = dict(free)
            forced = dict(free)
        else:
            horizon_days = max(1, math.ceil(self.horizon_turns / 24))
            visible = self._visible_supply(rival_farm(obs), int(obs.get("day") or 0), horizon_days)
            # Until the known production bill is funded, projected future
            # prices cannot justify withholding legally sellable stock.  A
            # hold is considered only using cash already on hand; forecast
            # arrivals are never counted as cash for a payment deadline.
            if cash + 1e-9 < operating_target:
                timing = {item: quantity for item, quantity in free.items() if quantity > 0}
                self.timing_details = {
                    item: {"reason": "OPERATING_CASH", "quantity": quantity,
                           "evaluation_mode": "cash_deadline_guard"}
                    for item, quantity in timing.items()
                }
            else:
                timing = {
                    item: self._timed_quantity(
                        obs, item, quantity, int(projected_arrivals.get(item) or 0), visible,
                    )
                    for item, quantity in free.items() if quantity > 0
                }
            timing = {item: quantity for item, quantity in timing.items() if quantity > 0}
            forced = self._subset_for_obligation(
                free, inventories, params, funding_gap, capacity_release, slots,
            )
            # A positive future budget is not itself a current shortfall.
            # Keep evaluating market opportunities once that budget is funded.
            # When it is not funded, the forced sale takes precedence and
            # timing-only sales can use any remaining legal order slots.
            quantities = {
                item: max(int(timing.get(item) or 0), int(forced.get(item) or 0))
                for item in SELLABLE
            }

        forced_items = {item for item, quantity in forced.items() if quantity > 0}
        ranked = []
        for item, quantity in quantities.items():
            quantity = min(int(free.get(item) or 0), max(0, int(quantity or 0)))
            if quantity <= 0:
                continue
            revenue, _ = sale_revenue(item, quantity, inventories[item], params)
            forced_rank = 1 if item in forced_items else 0
            ranked.append((forced_rank, revenue / quantity, revenue, item, quantity))
        ranked.sort(reverse=True)

        # Forced item types get the scarce order slots first.  Timing-only
        # sales use whatever remains and never displace an operating obligation.
        chosen = ranked[:slots]
        orders = [["SELL", item, quantity] for _forced, _unit, _revenue, item, quantity in chosen]
        expected = sum(row[2] for row in chosen)
        stressed = self.stressed_revenue(obs, orders)
        shortfall = max(0.0, funding_gap - stressed)
        self.own_last_sales = {str(order[1]): int(order[2]) for order in orders}
        return SellPlan(
            orders=orders,
            expected_revenue=expected,
            forced=bool(terminal or funding_gap > 0 or capacity_release > 0),
            funding_target=funding_target,
            funding_gap=funding_gap,
            funding_shortfall=shortfall,
            capacity_release=capacity_release,
            timing_quantities=timing,
            forced_quantities=forced,
            timing_details=dict(self.timing_details),
        )
