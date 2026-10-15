"""Semantic tokens for walking a featured seat day by day from the opening."""
from __future__ import annotations

from config import ANIMALS, CROPS


def band(n) -> str:
    """Original count bins. Nearby sizes are not merged."""
    n = int(n or 0)
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 9:
        return "6-9"
    if n <= 14:
        return "10-14"
    return "15+"


def crop_token(crops: dict | None) -> str:
    parts = []
    for name in CROPS:
        label = band((crops or {}).get(name))
        if label != "0":
            parts.append(f"{name[0]}{label}")
    return ",".join(parts) or "fallow"


def animal_token(animals: dict | None) -> str:
    parts = []
    for name in ANIMALS:
        label = band((animals or {}).get(name))
        if label != "0":
            parts.append(f"{name[0]}{label}")
    return ",".join(parts) or "none"


def shop_token(shops) -> str:
    """Town shops are observations, not own production decisions. Kept for diagnostics only."""
    shops = list(shops or [])
    if not shops:
        return "S0"
    tail = shops[-2:] if len(shops) >= 2 else shops
    return "S:" + ">".join(str(x) for x in tail)


def day_token(snap: dict) -> str:
    """Own production decisions: crops, herd, dawn workers, land. No shops, no sales."""
    return "|".join(
        (
            crop_token(snap.get("crops")),
            animal_token(snap.get("animals")),
            f"w{int(snap.get('workers') or 1)}",
            f"l{int(snap.get('land') or 1)}",
        )
    )


CROP_ZH = {"W": "麦", "C": "萝", "T": "茄", "S": "莓", "M": "瓜"}
ANIMAL_ZH = {"G": "鹅", "C": "牛", "S": "羊"}


def parse_bundle(chunk: str) -> dict[str, str]:
    if not chunk or chunk in ("fallow", "none", "?"):
        return {}
    out: dict[str, str] = {}
    for part in chunk.split(","):
        part = part.strip()
        if len(part) < 2:
            continue
        out[part[0]] = part[1:]
    return out


def zh_bundle(chunk: str, table: dict[str, str]) -> str:
    parsed = parse_bundle(chunk)
    if not parsed:
        if chunk in ("fallow", "none", "空", "", None):
            return "空"
        return chunk or "空"
    return " ".join(table.get(letter, letter) + band_label for letter, band_label in parsed.items()) or "空"


def zh_token(token: str) -> str:
    parts = (token or "").split("|")
    while len(parts) < 4:
        parts.append("?")
    crops, animals, workers, land = parts[0], parts[1], parts[2], parts[3]
    return f"{zh_bundle(crops, CROP_ZH)} · {zh_bundle(animals, ANIMAL_ZH)} · {workers}{land}"


def opening_root(daily: list[dict]) -> tuple[int, str]:
    """Tree root = first morning that already shows an opening dump (day 2 if present)."""
    if not daily:
        return 0, "empty"
    if len(daily) > 2:
        snap = daily[2]
        return int(snap.get("day") or 2), day_token(snap)
    snap = daily[-1]
    return int(snap.get("day") or 0), day_token(snap)


def walk_path(daily: list[dict]) -> list[tuple[int, str]]:
    """Emit a token only when the day's operating signature changes."""
    path: list[tuple[int, str]] = []
    prev = None
    start_index = 2 if len(daily) > 2 else 0
    for snap in daily[start_index:]:
        token = day_token(snap)
        day = int(snap.get("day") or 0)
        if token != prev:
            path.append((day, token))
            prev = token
    if not path and daily:
        snap = daily[start_index] if start_index < len(daily) else daily[0]
        path.append((int(snap.get("day") or 0), day_token(snap)))
    return path


def occupancy_vs_token(crops: dict | None, animals: dict | None, land, token: str) -> dict:
    """True when live occupancy falls in the same original bands as a forest token."""
    parts = (token or "").split("|")
    while len(parts) < 4:
        parts.append("?")
    want_c = parse_bundle(parts[0])
    want_a = parse_bundle(parts[1])
    want_land = 1
    if parts[3].startswith("l"):
        try:
            want_land = int(parts[3][1:] or 1)
        except ValueError:
            want_land = 1
    crop_hits = []
    for name in CROPS:
        live = band((crops or {}).get(name))
        want = want_c.get(name[0], "0")
        crop_hits.append(live == want)
    animal_hits = []
    for name in ANIMALS:
        live = band((animals or {}).get(name))
        want = want_a.get(name[0], "0")
        animal_hits.append(live == want)
    land_ok = int(land or 1) == int(want_land)
    n_c = sum(crop_hits)
    n_a = sum(animal_hits)
    return {
        "crops": n_c == len(CROPS),
        "animals": n_a == len(ANIMALS),
        "land": land_ok,
        "ok": n_c == len(CROPS) and n_a == len(ANIMALS) and land_ok,
        "crop_hits": n_c,
        "animal_hits": n_a,
    }
