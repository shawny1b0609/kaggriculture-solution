"""Small masked state-candidate scorer. Options framework: freeze micro ProgramVM, score forest children."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mining.features import BRANCH_DIM, MAX_CHOICES, OBS_DIM


class NodeScorer(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, branch_dim: int = BRANCH_DIM, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.branch_dim = branch_dim
        self.hidden = hidden
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.branch_net = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def logits(self, obs: torch.Tensor, branches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # obs: B,D  branches: B,K,Db  mask: B,K bool
        bsz, k, _ = branches.shape
        o = self.obs_net(obs).unsqueeze(1).expand(bsz, k, -1)
        b = self.branch_net(branches)
        scores = self.head(torch.cat([o, b], dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)

    def log_probs(self, obs, branches, mask) -> torch.Tensor:
        return F.log_softmax(self.logits(obs, branches, mask), dim=-1)


def pack_branches(vecs: list[list[float]], cap: int = MAX_CHOICES) -> tuple[list[list[float]], list[bool]]:
    padded = [list(v) for v in vecs[:cap]]
    mask = [True] * len(padded)
    while len(padded) < cap:
        padded.append([0.0] * BRANCH_DIM)
        mask.append(False)
    return padded, mask


def save_policy(model: NodeScorer, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "branch_dim": model.branch_dim,
        "hidden": int(getattr(model, "hidden", 64)),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_policy(path: Path, map_location="cpu") -> NodeScorer:
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    model = NodeScorer(int(blob["obs_dim"]), int(blob["branch_dim"]), int(blob.get("hidden") or 64))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
