"""Load episode JSON files and expose (state_tensor, MC_return) pairs.

Includes two pieces of data-efficiency machinery:

  - Potential-based reward shaping: the regression target at state s_t is
    G_t - Phi(s_t), where Phi has three terms: (1) quadratic row/column fill
    to reward near-complete lines; (2) transition penalty — total filled<->empty
    flips across all rows and columns, subtracted from Phi so fragmented boards
    (many interleaved filled/empty cells) are penalised; (3) piece fittability
    — sum of |p| * num_legal_placements(p, board) over all 32 piece types,
    directly penalising boards where large pieces are unplaceable.
    Original optimal policy is preserved iff the policy adds Phi(s') back at
    action-selection time (see `blockblaster/agent/policy.py`).

  - Dihedral (D4) symmetry augmentation: each (state, target) pair is
    expanded into 8 spatial variants (4 rotations x 2 reflections).  The
    game's value function is invariant under these transforms because the
    board has 4-fold rotational + reflection symmetry, so this is "free"
    training signal.  Enable / disable via `param.USE_DIHEDRAL_AUG`.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

import param
from blockblaster.game.board import Board
from blockblaster.game.pieces import PIECE_BY_ID, Piece
from blockblaster.game.potential import board_potential
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


def _dihedral_variants(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Return the 8 D4 symmetries of a (C, H, W) tensor.

    Applied identically across channels, which corresponds to rotating /
    reflecting the entire (board + piece-raster) input together.  The value
    function is invariant under this group, so the target is unchanged.
    """
    variants: list[torch.Tensor] = []
    for k in range(4):
        rot = torch.rot90(tensor, k, dims=(-2, -1))
        variants.append(rot.contiguous())
        variants.append(torch.flip(rot, dims=(-1,)).contiguous())
    return variants


class EpisodeDataset(Dataset):
    """
    Loads all episodes from `sim_dir`, computes shaped MC returns, and exposes
    (state_tensor, target) pairs for training.

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
        use_aug: bool | None = None,
    ) -> None:
        directory = sim_dir or param.SIMULATIONS_DIR
        test_frac = test_fraction if test_fraction is not None else param.TEST_SPLIT
        seed = split_seed if split_seed is not None else param.SPLIT_SEED
        gam = gamma if gamma is not None else param.GAMMA
        # Augmentation only on training data — leaks otherwise and inflates
        # apparent test-set size unhelpfully.
        augment = (use_aug if use_aug is not None else param.USE_DIHEDRAL_AUG) and split == "train"

        episode_paths = list_episodes(directory)
        if not episode_paths:
            raise FileNotFoundError(
                f"No episode files found in '{directory}'. "
                "Run simulate.py first."
            )

        rng = random.Random(seed)
        shuffled = list(episode_paths)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_frac))
        if split == "test":
            selected = shuffled[:n_test]
        else:
            selected = shuffled[n_test:]

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
                # Shaped target: G_t - Phi(s_t).  Phi(terminal) := 0 is implicit
                # because terminal states are not stored in the trajectory.
                target = ret - board_potential(board.grid)
                if augment:
                    for variant in _dihedral_variants(tensor):
                        self._items.append((variant, target))
                else:
                    self._items.append((tensor, target))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor, ret = self._items[idx]
        return tensor, torch.tensor(ret, dtype=torch.float32)
