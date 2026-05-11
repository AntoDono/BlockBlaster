"""Load episode JSON files and expose (state_tensor, MC_return) pairs."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

import param
from blockblaster.game.board import Board
from blockblaster.game.pieces import PIECE_BY_ID, Piece
from blockblaster.model.encoder import encode_state
from blockblaster.sim.io import list_episodes, read_episode


def _compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted MC returns G_t = sum_{k>=t} gamma^(k-t) * r_k."""
    returns: list[float] = [0.0] * len(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


class EpisodeDataset(Dataset):
    """
    Loads all episodes from `sim_dir`, computes MC returns, and exposes
    (state_tensor, return) pairs for training.

    The train/test split is done at the episode level (seeded) to prevent
    leakage between correlated successive states within a trajectory.
    """

    def __init__(
        self,
        sim_dir: str | None = None,
        split: str = "train",
        test_fraction: float | None = None,
        split_seed: int | None = None,
        gamma: float | None = None,
    ) -> None:
        directory = sim_dir or param.SIMULATIONS_DIR
        test_frac = test_fraction if test_fraction is not None else param.TEST_SPLIT
        seed = split_seed if split_seed is not None else param.SPLIT_SEED
        gam = gamma if gamma is not None else param.GAMMA

        episode_paths = list_episodes(directory)
        if not episode_paths:
            raise FileNotFoundError(
                f"No episode files found in '{directory}'. "
                "Run simulate.py first."
            )

        # Episode-level split (shuffled but reproducible)
        rng = random.Random(seed)
        shuffled = list(episode_paths)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_frac))
        if split == "test":
            selected = shuffled[:n_test]
        else:
            selected = shuffled[n_test:]

        # Build flat list of (state_tensor, return) pairs
        self._items: list[tuple[torch.Tensor, float]] = []
        for path in selected:
            episode = read_episode(path)
            rewards = [step["reward"] for step in episode["steps"]]
            returns = _compute_returns(rewards, gam)
            for step, ret in zip(episode["steps"], returns):
                board = Board.from_list(step["board"])
                queue: list[Piece] = [
                    PIECE_BY_ID[pid] for pid in step["queue"]
                    if pid in PIECE_BY_ID
                ]
                tensor = encode_state(board, queue)
                self._items.append((tensor, ret))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor, ret = self._items[idx]
        return tensor, torch.tensor(ret, dtype=torch.float32)
