"""Save and load ValueNet checkpoints.

Checkpoints may optionally persist Adam optimizer state (momentum / variance
estimates) across `train()` calls so we don't throw away learned curvature
information every round of the simulate -> train loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import param
from blockblaster.model.value_net import ValueNet


def save(
    net: ValueNet,
    epoch: int,
    best_test_loss: float,
    path: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a checkpoint, optionally including optimizer state."""
    ckpt_path = Path(path or param.CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "state_dict": net.state_dict(),
        "epoch": epoch,
        "best_test_loss": best_test_loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)


def load(net: ValueNet, path: str | None = None) -> dict:
    """Load checkpoint into `net` (in-place). Returns the full payload dict
    (which may include `optimizer_state` if it was saved)."""
    ckpt_path = Path(path or param.CHECKPOINT_PATH)
    data = torch.load(ckpt_path, map_location=param.DEVICE, weights_only=True)
    net.load_state_dict(data["state_dict"])
    return data


def load_if_exists(
    net: ValueNet, path: str | None = None
) -> Optional[dict]:
    """Load checkpoint if the file exists; return None otherwise."""
    ckpt_path = Path(path or param.CHECKPOINT_PATH)
    if not ckpt_path.exists():
        return None
    return load(net, str(ckpt_path))


def resolve_sim_checkpoint_path() -> Optional[Path]:
    """Return the checkpoint path simulation should load this round.

    Priority:
      1. BEST_CHECKPOINT_PATH — exists after the first round that improved
         the mean score.  Simulation always uses these stable best-mean weights
         so the sim policy never regresses even as training advances.
      2. CHECKPOINT_PATH — used on early rounds before any BEST snapshot exists.
      3. None — no checkpoint at all (round 1 cold-start → random policy).
    """
    best = Path(param.BEST_CHECKPOINT_PATH)
    if best.exists():
        return best
    latest = Path(param.CHECKPOINT_PATH)
    if latest.exists():
        return latest
    return None
