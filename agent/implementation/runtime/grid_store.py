"""Dawn-grid net: same tile layout at the same day is one node.

Edges are next-day grids only. Shopping is derived after a path is chosen.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config import GRID_EDGES, GRID_NODES, GRID_WEIGHTS
from mining.features import _card_from_counts
from mining.grid_key import materials_needed


class GridNet:
    def __init__(self, nodes: dict[int, dict], by_key: dict[tuple, int], outgoing: dict[int, list[dict]], weights: dict[str, float] | None = None):
        self.nodes = nodes
        self.by_key = by_key
        self.outgoing = outgoing
        self._day0 = next((int(n["id"]) for n in nodes.values() if int(n["day"]) == 0), 0)
        self.weights = dict(weights or self._equal_roots())

    def _equal_roots(self) -> dict[str, float]:
        """Equal per featured seat through an opening, not per unique grid."""
        return {str(int(edge["dst"])): float(edge.get("n") or 1) for edge in self.outgoing.get(self._day0, [])}

    def day0_id(self) -> int:
        return int(self._day0)

    def root_edges(self) -> list[dict]:
        return list(self.outgoing.get(self._day0) or [])

    @classmethod
    def load(cls, nodes_path: Path | None = None, edges_path: Path | None = None) -> "GridNet":
        nodes_path = nodes_path or GRID_NODES
        edges_path = edges_path or GRID_EDGES
        nodes: dict[int, dict] = {}
        by_key: dict[tuple, int] = {}
        with nodes_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                nid = int(row["id"])
                nodes[nid] = row
                by_key[(int(row["day"]), str(row["grid"]))] = nid
        outgoing: dict[int, list[dict]] = defaultdict(list)
        if edges_path.exists():
            with edges_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    outgoing[int(row["src"])].append(row)
        for src in outgoing:
            outgoing[src].sort(key=lambda r: -int(r.get("n") or 0))
        net = cls(nodes, by_key, dict(outgoing))
        if GRID_WEIGHTS.exists():
            try:
                blob = json.loads(GRID_WEIGHTS.read_text(encoding="utf-8"))
                if isinstance(blob, dict) and blob:
                    merged = net._equal_roots()
                    merged.update({str(k): float(v) for k, v in blob.items()})
                    net.weights = merged
            except Exception:
                pass
        return net

    def save_weights(self, path: Path | None = None) -> None:
        path = path or GRID_WEIGHTS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.weights, indent=2, ensure_ascii=False), encoding="utf-8")

    def lookup(self, day: int, ghash: str) -> dict | None:
        nid = self.by_key.get((int(day), str(ghash)))
        if nid is None:
            return None
        return self.nodes.get(nid)

    def best_edge(self, node_id: int) -> dict | None:
        rows = self.outgoing.get(int(node_id)) or []
        return rows[0] if rows else None

    def dest_of(self, edge: dict | None) -> dict | None:
        if not edge:
            return None
        return self.nodes.get(int(edge["dst"]))

    def card_for(self, dest: dict | None, stats: dict | None, private: dict | None, program_id: str) -> dict:
        occup = dest or {}
        card = _card_from_counts(
            occup.get("crops") or {},
            occup.get("animals") or {},
            int(occup.get("land") or 1),
            program_id,
        )
        if dest is not None and stats is not None:
            card["day_schedule"] = materials_needed(stats, private or {}, dest)
        return card
