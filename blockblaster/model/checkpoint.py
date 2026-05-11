"""Save and load ValueNet checkpoints."""

from __future__ import annotations

import os
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
) -> None:
    ckpt_path = Path(path or param.CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "epoch": epoch,
            "best_test_loss": best_test_loss,
        },
        ckpt_path,
    )


def load(net: ValueNet, path: str | None = None) -> dict:
    """Load checkpoint into `net` (in-place). Returns the metadata dict."""
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
