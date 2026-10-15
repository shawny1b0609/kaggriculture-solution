"""Locate featured replays: loose JSON, ZIP members, JSON.GZ. Dedup by episode id."""
from __future__ import annotations

import csv
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import CUTOFF_DATE, FEATURED, FEATURED_INDEX, FEATURED_RECENT


@dataclass(frozen=True)
class Locator:
    episode_id: str
    date: str
    kind: str  # json | zip | gz
    path: str
    member: str
    avg_score: float
    size_bytes: int


def index_expected() -> dict[str, int]:
    out = {}
    with FEATURED_INDEX.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["date"] <= CUTOFF_DATE:
                out[row["date"]] = int(row["episode_count"])
    return out


def _manifest_rows(folder: Path) -> list[dict]:
    path = folder / "manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def day_jobs(cutoff: str = CUTOFF_DATE) -> list[dict]:
    jobs = []
    expected = index_expected()
    for date in sorted(expected):
        if date > cutoff:
            continue
        folder = FEATURED / date
        if not folder.is_dir():
            jobs.append({"date": date, "kind": "missing_dir", "expected": expected[date], "rows": [], "folder": str(folder)})
            continue
        rows = _manifest_rows(folder)
        archive = folder / f"kaggriculture-episodes-{date}.zip"
        kind = "zip" if archive.exists() else "json"
        items = []
        for row in rows:
            items.append(
                {
                    "episode_id": str(row["episode_id"]),
                    "avg_score": float(row.get("avg_score") or 0),
                    "size_bytes": int(row.get("size_bytes") or 0),
                }
            )
        jobs.append(
            {
                "date": date,
                "kind": kind,
                "expected": expected[date],
                "n_manifest": len(items),
                "archive": str(archive) if archive.exists() else "",
                "folder": str(folder),
                "rows": items,
            }
        )
    return jobs


def zip_member_map(archive: Path) -> dict[str, str]:
    names = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or not info.filename.endswith(".json"):
                continue
            names[Path(info.filename).stem] = info.filename
    return names


def read_bytes(job: dict, episode_id: str, members: dict[str, str] | None = None) -> bytes | None:
    if job["kind"] == "zip":
        archive = Path(job["archive"])
        mapping = members if members is not None else zip_member_map(archive)
        name = mapping.get(str(episode_id))
        if not name:
            return None
        with zipfile.ZipFile(archive) as bundle:
            return bundle.read(name)
    path = Path(job["folder"]) / f"{episode_id}.json"
    if path.exists():
        return path.read_bytes()
    gz = Path(job["folder"]) / f"{episode_id}.json.gz"
    if gz.exists():
        import gzip

        return gzip.decompress(gz.read_bytes())
    recent = FEATURED_RECENT / job["date"] / f"{episode_id}.json.gz"
    if recent.exists():
        import gzip

        return gzip.decompress(recent.read_bytes())
    return None
