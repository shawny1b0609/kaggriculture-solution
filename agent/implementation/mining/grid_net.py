"""Scan featured seats day-by-day. Identical dawn grids merge; edges are next grids.

The net is only state transitions. Shopping is derived after a day's path is
chosen, not stored as the edge identity.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import orjson

from config import ENGINE_VERSION, GRAPHS, REPORTS, STATES
from corpus.catalog import day_jobs, read_bytes, zip_member_map
from corpus.extract import _obs
from mining.grid_key import exact_hash, occup_from_farm


def walk_dawns(steps: list, seat: int):
    """Yield (day, farm) at each hour-0 observation."""
    if not steps:
        return
    prev = _obs(steps[0], seat, None)
    if prev is None:
        return
    farms = prev.get("farms") or [{}, {}]
    if seat >= len(farms):
        return
    yield 0, farms[seat]
    for t in range(1, len(steps)):
        obs = _obs(steps[t], seat, prev)
        hour = int(obs.get("hour") or 0)
        day = int(obs.get("day") or 0)
        farm = (obs.get("farms") or [{}, {}])[seat]
        if hour == 0 and t > 1:
            yield day, farm
        prev = obs


def _seat_ok(replay: dict, seat: int) -> dict | None:
    version = str(replay.get("module_version") or "")
    steps = replay.get("steps") or []
    cfg = replay.get("configuration") or {}
    complete = len(steps) >= STATES
    legal = complete and version == ENGINE_VERSION and int(cfg.get("episodeSteps") or 0) in (0, STATES)
    if not legal:
        return None
    rewards = [float(x) if x is not None else None for x in (replay.get("rewards") or [None, None])[:2]]
    while len(rewards) < 2:
        rewards.append(None)
    win = False
    if rewards[0] is not None and rewards[1] is not None:
        win = (seat == 0 and rewards[0] > rewards[1]) or (seat == 1 and rewards[1] > rewards[0])
    return {
        "reward": rewards[seat],
        "rival_reward": rewards[1 - seat],
        "win": win,
        "late_cash": float(rewards[seat] or 0),
        "steps": steps,
    }


def mine_job(job: dict, smoke: int = 0) -> dict:
    members = None
    if job.get("kind") == "zip" and job.get("archive"):
        members = zip_member_map(Path(job["archive"]))
    snaps: dict[tuple, dict] = {}
    edges: dict[tuple, dict] = {}
    seen = 0
    used = 0
    skipped = 0
    rows = job.get("rows") or []
    for row in rows:
        seen += 1
        payload = read_bytes(job, row["episode_id"], members)
        if payload is None:
            skipped += 1
            continue
        replay = orjson.loads(payload)
        for seat in (0, 1):
            packed = _seat_ok(replay, seat)
            if packed is None:
                continue
            used += 1
            loc = f"{job['date']}:{row['episode_id']}:{seat}"
            win = 1 if packed.get("win") else 0
            cash = float(packed.get("late_cash") or 0)
            prev = None
            for day, farm in walk_dawns(packed["steps"], seat):
                ghash = exact_hash(farm)
                key = (int(day), ghash)
                node = snaps.get(key)
                if node is None:
                    occup = occup_from_farm(farm)
                    snaps[key] = {
                        "n": 1,
                        "wins": win,
                        "cash_sum": cash,
                        "loc": loc,
                        "occup": occup,
                    }
                else:
                    node["n"] += 1
                    node["wins"] += win
                    node["cash_sum"] += cash
                if prev is not None:
                    prev_day, prev_hash = prev
                    ekey = (prev_day, prev_hash, int(day), ghash)
                    edge = edges.get(ekey)
                    if edge is None:
                        edges[ekey] = {"n": 1, "wins": win, "cash_sum": cash, "loc": loc}
                    else:
                        edge["n"] += 1
                        edge["wins"] += win
                        edge["cash_sum"] += cash
                prev = (int(day), ghash)
    return {"date": job["date"], "seen": seen, "used": used, "skipped": skipped, "snaps": snaps, "edges": edges}


def _merge(parts: list[dict]) -> dict:
    nodes: dict[tuple, dict] = {}
    nid = 0
    for part in parts:
        for (day, ghash), rec in (part.get("snaps") or {}).items():
            key = (int(day), ghash)
            node = nodes.get(key)
            if node is None:
                occup = rec["occup"]
                nodes[key] = {
                    "id": nid,
                    "day": int(day),
                    "grid": ghash,
                    "n": int(rec["n"]),
                    "wins": int(rec["wins"]),
                    "cash_sum": float(rec["cash_sum"]),
                    "example": rec["loc"],
                    "crops": occup["crops"],
                    "animals": occup["animals"],
                    "land": occup["land"],
                }
                nid += 1
            else:
                node["n"] += int(rec["n"])
                node["wins"] += int(rec["wins"])
                node["cash_sum"] += float(rec["cash_sum"])
    edges: dict[tuple, dict] = {}
    for part in parts:
        for (pday, phash, day, ghash), rec in (part.get("edges") or {}).items():
            src = nodes.get((int(pday), phash))
            dst = nodes.get((int(day), ghash))
            if not src or not dst:
                continue
            key = (src["id"], dst["id"])
            edge = edges.get(key)
            if edge is None:
                edges[key] = {
                    "src": src["id"],
                    "dst": dst["id"],
                    "n": int(rec["n"]),
                    "wins": int(rec["wins"]),
                    "cash_sum": float(rec["cash_sum"]),
                    "example": rec["loc"],
                }
            else:
                edge["n"] += int(rec["n"])
                edge["wins"] += int(rec["wins"])
                edge["cash_sum"] += float(rec["cash_sum"])
    node_rows = []
    for node in nodes.values():
        n = max(1, int(node["n"]))
        node_rows.append(
            {
                "id": node["id"],
                "day": node["day"],
                "grid": node["grid"],
                "n": node["n"],
                "wins": node["wins"],
                "mean_cash": round(node["cash_sum"] / n, 1),
                "example": node["example"],
                "crops": node["crops"],
                "animals": node["animals"],
                "land": node["land"],
                "degree": 0,
            }
        )
    node_rows.sort(key=lambda r: r["id"])
    by_src = Counter()
    edge_rows = []
    for edge in edges.values():
        n = max(1, int(edge["n"]))
        edge_rows.append(
            {
                "src": edge["src"],
                "dst": edge["dst"],
                "n": edge["n"],
                "wins": edge["wins"],
                "mean_cash": round(edge["cash_sum"] / n, 1),
                "example": edge["example"],
            }
        )
        by_src[edge["src"]] += 1
    for row in node_rows:
        row["degree"] = int(by_src.get(row["id"]) or 0)
    edge_rows.sort(key=lambda r: (-r["n"], r["src"], r["dst"]))
    return node_rows, edge_rows


def build(smoke: int = 0, limit_days: int | None = None, workers: int = 1, min_date: str = "2026-08-15") -> dict:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    jobs = [job for job in day_jobs() if not min_date or job["date"] >= min_date]
    if limit_days:
        jobs = jobs[:limit_days]
    if smoke:
        jobs = [dict(job, rows=(job.get("rows") or [])[:smoke]) for job in jobs]
        jobs = [job for job in jobs if job["rows"]]
    parts = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            part = mine_job(job)
            print({"date": part["date"], "used": part["used"], "grids": len(part["snaps"])}, flush=True)
            parts.append(part)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(mine_job, job): job["date"] for job in jobs}
            for fut in as_completed(futures):
                part = fut.result()
                print({"date": part["date"], "used": part["used"], "grids": len(part["snaps"])}, flush=True)
                parts.append(part)
    node_rows, edge_rows = _merge(parts)
    branch = sum(1 for row in node_rows if int(row["degree"]) > 1)
    unary = sum(1 for row in node_rows if int(row["degree"]) == 1)
    leaves = sum(1 for row in node_rows if int(row["degree"]) == 0)
    used = sum(int(p.get("used") or 0) for p in parts)
    snaps = sum(sum(int(rec["n"]) for rec in (p.get("snaps") or {}).values()) for p in parts)
    by_day = Counter(row["day"] for row in node_rows)
    summary = {
        "kind": "dawn_grid_net",
        "step": "day_hour0",
        "merge": "exact_tile_state_same_day",
        "edges": "state_transition_only",
        "shopping": "derived_after_path_choice",
        "seats_used": used,
        "snapshots": snaps,
        "n_nodes": len(node_rows),
        "n_edges": len(edge_rows),
        "n_branch": branch,
        "n_unary": unary,
        "n_leaves": leaves,
        "mean_share": round(snaps / max(1, len(node_rows)), 2),
        "nodes_by_day": {str(k): int(by_day[k]) for k in sorted(by_day)},
        "files": {
            "nodes": "artifacts/graphs/grid_net_nodes.jsonl",
            "edges": "artifacts/graphs/grid_net_edges.jsonl",
        },
        "do_not_overwrite": ["path_forest_close_nodes.jsonl", "path_forest_nodes.jsonl"],
    }
    (GRAPHS / "grid_net_nodes.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in node_rows) + "\n",
        encoding="utf-8",
    )
    (GRAPHS / "grid_net_edges.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in edge_rows) + "\n",
        encoding="utf-8",
    )
    (REPORTS / "grid_net.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
