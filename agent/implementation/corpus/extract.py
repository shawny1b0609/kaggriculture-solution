"""Full-body dual-seat event extraction. One jsonl.gz row per seat per episode."""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import orjson

from config import ANIMALS, CROPS, EVENTS, REPORTS, STATES
from corpus.catalog import day_jobs, read_bytes, zip_member_map

STEPS_DAY = 24
CHECKPOINTS = (0, 24, 72, 144, 240, 360, 480, 600, 719)


def _tile_label(tile) -> str:
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED" or tile == "WEED":
        return str(tile)
    if not isinstance(tile, dict):
        return str(tile)
    kind = tile.get("kind") or "EMPTY"
    if kind == "PLANT":
        return "PLANT:" + str(tile.get("crop") or "?")
    if kind in ("COOP", "PASTURE"):
        return kind + ":" + str(tile.get("animal") or "EMPTY")
    if kind == "WEED":
        return "WEED"
    return str(kind)


def farm_summary(farm: dict) -> dict:
    crops = {name: 0 for name in CROPS}
    animals = {name: 0 for name in ANIMALS}
    empty = weeds = ready = hungry = 0
    for row in farm.get("tiles") or []:
        for tile in row or []:
            if tile is None:
                empty += 1
                continue
            if tile == "LOCKED":
                continue
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds += 1
                continue
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in crops:
                    crops[crop] += 1
                if int(tile.get("yield_units") or 0) > 0:
                    ready += 1
            elif isinstance(tile, dict) and tile.get("animal"):
                animal = tile.get("animal")
                if animal in animals:
                    animals[animal] += 1
                if not tile.get("fed_today"):
                    hungry += 1
                if int(tile.get("yield_units") or 0) > 0:
                    ready += 1
    return {
        "cash": float(farm.get("money") or 0),
        "workers": 1 + len(farm.get("hands") or []),
        "land": len(farm.get("unlocked_quadrants") or []),
        "crops": crops,
        "animals": animals,
        "empty": empty,
        "weeds": weeds,
        "ready": ready,
        "hungry": hungry,
    }


def _obs(frame, seat: int, prev):
    if not isinstance(frame, list) or seat >= len(frame):
        return prev
    cell = frame[seat] or {}
    obs = cell.get("observation")
    if obs:
        return obs
    if seat != 0 and isinstance(frame[0], dict) and frame[0].get("observation"):
        shared = frame[0]["observation"]
        if prev is None:
            return shared
        merged = dict(prev)
        for key in ("farms", "market", "town", "day", "hour", "step"):
            if key in shared:
                merged[key] = shared[key]
        return merged
    return prev


def _action(frame, seat: int):
    if not isinstance(frame, list) or seat >= len(frame):
        return None
    return (frame[seat] or {}).get("action")


def _market_orders(action) -> list:
    if not isinstance(action, dict):
        return []
    out = []
    for order in action.get("market") or []:
        if isinstance(order, list) and order:
            out.append([str(order[0]), *[x for x in order[1:]]])
    return out


def _field_ops(action) -> list[str]:
    if not isinstance(action, dict):
        return []
    ops = []
    farmer = action.get("farmer") or ["PASS"]
    if isinstance(farmer, list) and farmer:
        ops.append(str(farmer[0]))
    for hand in action.get("hands") or []:
        if isinstance(hand, list) and hand:
            ops.append(str(hand[0]))
    return ops


def classify_family(open_crops: dict, open_animals: dict, late_animals: dict, late_crops: dict) -> str:
    cows = int(late_animals.get("COW") or 0)
    sheep = int(late_animals.get("SHEEP") or 0)
    geese = int(late_animals.get("GOOSE") or 0)
    livestock = cows + sheep + geese
    melon = int(late_crops.get("MELON") or 0)
    wheat = int(late_crops.get("WHEAT") or 0)
    tomato = int(late_crops.get("TOMATO") or 0)
    berry = int(late_crops.get("STRAWBERRY") or 0)
    carrot = int(late_crops.get("CARROT") or 0)
    if livestock >= 6:
        if sheep >= cows and sheep >= geese and sheep >= 4:
            return "sheep_heavy"
        if geese >= cows and geese >= sheep and geese >= 6:
            return "goose_heavy"
        if cows >= 6:
            return "cow_heavy"
        return "mixed_livestock"
    if melon >= 8 and livestock <= 2:
        return "melon_field"
    if tomato + berry >= 8 and livestock <= 2:
        return "ongoing_crops"
    if carrot >= 10 and livestock <= 2:
        return "carrot_cycle"
    if wheat >= 10 and livestock <= 2:
        return "wheat_cycle"
    if livestock:
        return "light_livestock"
    return "other"


def extract_seat(steps: list, seat: int) -> dict:
    n = len(steps)
    prev = _obs(steps[0], seat, None)
    if prev is None:
        return {"error": "no_initial_obs"}
    farms = prev.get("farms") or [{}, {}]
    if seat >= len(farms):
        return {"error": "missing_farm"}
    daily = []
    events = []
    plants_req = Counter()
    plants_act = Counter()
    animals_req = Counter()
    animals_act = Counter()
    buys = Counter()
    sells = Counter()
    ops = Counter()
    hires = 0
    lands = 0
    first_plant_t = None
    first_animal_t = None
    first_hire_t = None
    first_land_t = None
    shop_hist = []
    prev_shops = list((prev.get("town") or {}).get("unlocked_shops") or [])
    prev_farm = farms[seat]
    prev_private = prev.get("private") or {}
    prev_seeds = dict(prev_private.get("seeds") or {})
    prev_shed = dict(prev_private.get("shed") or {})
    checkpoints = {}
    rival_hist = []

    def snap(obs, farm):
        summary = farm_summary(farm)
        rival = farm_summary((obs.get("farms") or [{}, {}])[1 - seat] if len(obs.get("farms") or []) > 1 else {})
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        return {
            "step": int(obs.get("step") or 0),
            "day": int(obs.get("day") or 0),
            "hour": int(obs.get("hour") or 0),
            "own": summary,
            "rival": {"crops": rival["crops"], "animals": rival["animals"], "workers": rival["workers"], "land": rival["land"], "cash": rival["cash"]},
            "shops": shops,
        }

    checkpoints["0"] = snap(prev, prev_farm)
    daily.append({"day": 0, **checkpoints["0"]["own"], "shops": checkpoints["0"]["shops"]})

    for t in range(1, n):
        action = _action(steps[t], seat)
        obs = _obs(steps[t], seat, prev)
        farm = (obs.get("farms") or [{}, {}])[seat]
        private = obs.get("private") or prev_private
        shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
        if shops != prev_shops:
            added = shops[len(prev_shops) :] if shops[: len(prev_shops)] == prev_shops else shops
            events.append({"t": t - 1, "e": "SHOP", "shops": added})
            shop_hist.append({"t": t - 1, "shops": shops})
            prev_shops = shops
        field_ops = _field_ops(action)
        for op in field_ops:
            ops[op] += 1
        if isinstance(action, dict):
            farmer = action.get("farmer") or []
            if isinstance(farmer, list) and farmer and farmer[0] == "PLANT" and len(farmer) > 1:
                plants_req[str(farmer[1])] += 1
            for hand in action.get("hands") or []:
                if isinstance(hand, list) and hand and hand[0] == "PLANT" and len(hand) > 1:
                    plants_req[str(hand[1])] += 1
        for order in _market_orders(action):
            kind = order[0]
            if kind == "HIRE":
                hires += 1
                if first_hire_t is None:
                    first_hire_t = t - 1
                events.append({"t": t - 1, "e": "HIRE"})
            elif kind == "BUY_LAND":
                lands += 1
                if first_land_t is None:
                    first_land_t = t - 1
                events.append({"t": t - 1, "e": "LAND"})
            elif kind in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL") and len(order) >= 3:
                item = str(order[1])
                qty = int(order[2] or 0)
                key = kind + ":" + item
                if kind == "SELL":
                    sells[item] += qty
                else:
                    buys[key] += qty
                    if kind == "BUY_ANIMAL":
                        animals_req[item] += qty
                events.append({"t": t - 1, "e": kind, "item": item, "n": qty})
        # Actual plants / animals by tile diff.
        old_tiles = prev_farm.get("tiles") or []
        new_tiles = farm.get("tiles") or []
        for y, row in enumerate(new_tiles):
            old_row = old_tiles[y] if y < len(old_tiles) else []
            for x, tile in enumerate(row or []):
                old = old_row[x] if x < len(old_row) else None
                new_l = _tile_label(tile)
                old_l = _tile_label(old)
                if new_l.startswith("PLANT:") and new_l != old_l:
                    crop = new_l.split(":", 1)[1]
                    plants_act[crop] += 1
                    if first_plant_t is None:
                        first_plant_t = t - 1
                if (new_l.startswith("PASTURE:") or new_l.startswith("COOP:")) and new_l != old_l:
                    animal = new_l.split(":", 1)[1]
                    if animal != "EMPTY":
                        animals_act[animal] += 1
                        if first_animal_t is None:
                            first_animal_t = t - 1
        hour = int(obs.get("hour") or 0)
        if hour == 0 and t > 1:
            daily.append({"day": int(obs.get("day") or 0), **farm_summary(farm), "shops": shops})
            rival = farm_summary((obs.get("farms") or [{}, {}])[1 - seat] if len(obs.get("farms") or []) > 1 else {})
            rival_hist.append({"day": int(obs.get("day") or 0), "animals": rival["animals"], "crops": rival["crops"]})
        if t in CHECKPOINTS or (t == n - 1):
            checkpoints[str(t)] = snap(obs, farm)
        prev = obs
        prev_farm = farm
        prev_private = private

    late = daily[-1] if daily else farm_summary(prev_farm)
    open_state = daily[2] if len(daily) > 2 else daily[0]
    family = classify_family(open_state.get("crops") or {}, open_state.get("animals") or {}, late.get("animals") or {}, late.get("crops") or {})
    return {
        "n_frames": n,
        "ops": dict(ops),
        "plants_requested": dict(plants_req),
        "plants_actual": dict(plants_act),
        "animals_requested": dict(animals_req),
        "animals_actual": dict(animals_act),
        "buys": dict(buys),
        "sells": dict(sells),
        "hires": hires,
        "lands": lands,
        "first_plant_t": first_plant_t,
        "first_animal_t": first_animal_t,
        "first_hire_t": first_hire_t,
        "first_land_t": first_land_t,
        "shop_events": shop_hist,
        "daily": daily,
        "checkpoints": checkpoints,
        "rival_daily": rival_hist,
        "family": family,
        "open_crops": open_state.get("crops") or {},
        "open_animals": open_state.get("animals") or {},
        "late_crops": late.get("crops") or {},
        "late_animals": late.get("animals") or {},
        "late_cash": late.get("cash"),
        "event_count": len(events),
        "events_head": events[:80],
    }


def extract_replay(raw: bytes, date: str, episode_id: str, avg_score: float) -> dict:
    replay = orjson.loads(raw)
    version = str(replay.get("module_version") or "")
    steps = replay.get("steps") or []
    cfg = replay.get("configuration") or {}
    rewards = [float(x) for x in (replay.get("rewards") or [None, None])[:2]]
    while len(rewards) < 2:
        rewards.append(None)
    teams = list((replay.get("info") or {}).get("TeamNames") or ["", ""])
    statuses = []
    if steps:
        last = steps[-1]
        for seat in (0, 1):
            if isinstance(last, list) and seat < len(last) and isinstance(last[seat], dict):
                statuses.append(str(last[seat].get("status") or ""))
            else:
                statuses.append("")
    complete = len(steps) >= STATES
    legal = complete and version == "1.32.7" and int(cfg.get("episodeSteps") or 0) in (0, STATES)
    rows = []
    errors = []
    for seat in (0, 1):
        try:
            payload = extract_seat(steps, seat) if complete else {"error": f"truncated_{len(steps)}"}
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            errors.append(seat)
        win = False
        draw = False
        if rewards[0] is not None and rewards[1] is not None:
            draw = rewards[0] == rewards[1]
            win = (seat == 0 and rewards[0] > rewards[1]) or (seat == 1 and rewards[1] > rewards[0])
        rows.append(
            {
                "date": date,
                "episode": episode_id,
                "seat": seat,
                "module_version": version,
                "n_steps": len(steps),
                "complete": complete,
                "eligible": bool(legal and not payload.get("error")),
                "reward": rewards[seat],
                "rival_reward": rewards[1 - seat],
                "win": win,
                "draw": draw,
                "team": teams[seat] if seat < len(teams) else "",
                "rival_team": teams[1 - seat] if 1 - seat < len(teams) else "",
                "avg_score": avg_score,
                "status": statuses[seat] if seat < len(statuses) else "",
                **payload,
            }
        )
    return {
        "episode": episode_id,
        "date": date,
        "module_version": version,
        "n_steps": len(steps),
        "eligible": legal,
        "rows": rows,
        "errors": errors,
    }


def parse_snap(path: Path) -> int:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                n += 1
    return n


def write_ok(path: Path, meta: dict) -> None:
    path.with_suffix(".ok").write_text(json.dumps(meta), encoding="utf-8")


def cache_ok(job: dict, out: Path) -> dict | None:
    expected = 2 * len(job["rows"])
    min_seats = max(2, int(0.95 * expected)) if expected else 0
    ok_path = out.with_suffix(".ok")
    if out.exists() and ok_path.exists():
        try:
            meta = json.loads(ok_path.read_text(encoding="utf-8"))
            if int(meta.get("size") or 0) == out.stat().st_size and int(meta.get("seats") or 0) >= min_seats:
                return {
                    "day": job["date"],
                    "kind": job["kind"],
                    "parsed_seats": int(meta["seats"]),
                    "eligible_seats": int(meta.get("eligible_seats") or 0),
                    "skipped": int(meta.get("skipped") or 0),
                    "errors": int(meta.get("errors") or 0),
                    "path": str(out),
                    "cached": True,
                }
        except Exception:
            ok_path.unlink(missing_ok=True)
    return None


def mine_day(job: dict) -> dict:
    EVENTS.mkdir(parents=True, exist_ok=True)
    out = EVENTS / f"{job['date']}.jsonl.gz"
    cached = cache_ok(job, out)
    if cached:
        return cached
    parsed = 0
    eligible = 0
    skipped = 0
    errors = 0
    versions = Counter()
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    members = zip_member_map(Path(job["archive"])) if job["kind"] == "zip" and job["archive"] else None
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as writer:
            for index, row in enumerate(job["rows"], 1):
                try:
                    payload = read_bytes(job, row["episode_id"], members)
                    if payload is None:
                        skipped += 1
                        continue
                    packed = extract_replay(payload, job["date"], row["episode_id"], row["avg_score"])
                    versions[packed["module_version"]] += 1
                    for item in packed["rows"]:
                        writer.write(orjson.dumps(item) + b"\n")
                        parsed += 1
                        if item.get("eligible"):
                            eligible += 1
                    errors += len(packed["errors"])
                except Exception:
                    errors += 1
                if index % 40 == 0:
                    print({"day": job["date"], "progress": index, "parsed_seats": parsed, "errors": errors}, flush=True)
        raw.flush()
        os.fsync(raw.fileno())
    try:
        seats = parse_snap(tmp)
        if seats != parsed:
            raise ValueError(f"validated {seats} != written {parsed}")
        tmp.replace(out)
        meta = {
            "day": job["date"],
            "seats": seats,
            "eligible_seats": eligible,
            "skipped": skipped,
            "errors": errors,
            "size": out.stat().st_size,
            "versions": dict(versions),
        }
        write_ok(out, meta)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return {"day": job["date"], "kind": job["kind"], "parsed_seats": 0, "skipped": skipped, "errors": errors + 1, "validate_error": str(exc)}
    return {
        "day": job["date"],
        "kind": job["kind"],
        "parsed_seats": parsed,
        "eligible_seats": eligible,
        "skipped": skipped,
        "errors": errors,
        "versions": dict(versions),
        "path": str(out),
    }


def main(workers: int = 4, limit_days: int | None = None, smoke: int = 0) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = day_jobs()
    if smoke:
        trimmed = []
        remaining = smoke
        for job in jobs:
            take = min(len(job["rows"]), max(1, remaining))
            job = dict(job)
            job["rows"] = job["rows"][:take]
            trimmed.append(job)
            remaining -= take
            if remaining <= 0:
                break
        jobs = trimmed
    if limit_days:
        jobs = jobs[:limit_days]
    reports = []
    print(f"days={len(jobs)} workers={workers} smoke={smoke}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(mine_day, job): job["date"] for job in jobs}
        for future in as_completed(futures):
            day = futures[future]
            try:
                report = future.result()
            except Exception as exc:
                report = {"day": day, "error": str(exc)}
            reports.append(report)
            print(report, flush=True)
    reports.sort(key=lambda row: row.get("day") or "")
    summary = {
        "days": len(reports),
        "parsed_seats": sum(int(row.get("parsed_seats") or 0) for row in reports),
        "eligible_seats": sum(int(row.get("eligible_seats") or 0) for row in reports),
        "skipped": sum(int(row.get("skipped") or 0) for row in reports),
        "errors": sum(int(row.get("errors") or 0) for row in reports),
        "cached": sum(1 for row in reports if row.get("cached")),
        "rows": reports,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "extract_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("days", "parsed_seats", "eligible_seats", "skipped", "errors", "cached")}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-days", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0, help="episode cap for reader debug, then full run required")
    args = parser.parse_args()
    main(workers=args.workers, limit_days=args.limit_days or None, smoke=args.smoke)
