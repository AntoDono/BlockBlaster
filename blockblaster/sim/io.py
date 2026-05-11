"""JSON serialization for episode trajectories."""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_episode(trajectory: dict, directory: str, episode_index: int) -> Path:
    """
    Write `trajectory` to `directory/ep_<timestamp>_<index>.json`.
    Creates the directory if it doesn't exist.
    Returns the path written.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    filename = out_dir / f"ep_{ts}_{episode_index:06d}.json"
    with open(filename, "w") as f:
        json.dump(trajectory, f, separators=(",", ":"))
    return filename


def read_episode(path: str | Path) -> dict:
    """Load a single episode trajectory from a JSON file."""
    with open(path) as f:
        return json.load(f)


def list_episodes(directory: str | Path) -> list[Path]:
    """Return sorted list of all episode JSON files in `directory`."""
    return sorted(Path(directory).glob("ep_*.json"))
