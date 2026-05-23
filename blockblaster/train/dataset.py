"""Load episode JSON files and expose n-step TD samples for training.

Each `__getitem__` returns six items per step `t`:

    (s_t, s_next, n_step_reward_sum, phi_t, phi_next, bootstrap_flag)

The trainer turns these into a shaped target

    target_F(s_t) = n_step_sum + bootstrap * γ^n * (V_target(s_next) + phi_next) - phi_t

so the net's output keeps its meaning as the *shaped* value V_F = V* - Φ.
When `bootstrap=0` (episode ended within n steps from t), the term containing
V_target vanishes and the target reduces to the pure MC return minus φ(s_t).

Why store states by reference (indices into a unique-tensor pool) instead of
pre-expanding the 8 dihedral variants like the previous MC dataset did:

  - With n-step TD we need s_t AND s_{t+n} per sample.  Pre-expanding 8×
    would roughly double memory vs. the old dataset.
  - Within an episode the same encoded state appears many times across
    consecutive items (state t+n of step t equals state t of step (t+n)),
    so a unique-tensor pool collapses that duplication for free.
  - Augmentation is applied lazily in __getitem__ by rotating both s_t and
    s_{t+n} with the same group element, so a sample's target remains
    well-defined (V_F is approximately D4-invariant).

Phi (board_potential) is approximately D4-invariant; we cache one phi per
unique state regardless of the rotation index.
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


# Rotation/reflection index 0..7 in the D4 group.  `rot` is the number of
# 90-degree rotations; `flip` indicates a final horizontal reflection.
def _apply_dihedral(tensor: torch.Tensor, idx: int) -> torch.Tensor:
    """Apply the `idx`-th D4 element to a (C, H, W) tensor."""
    if idx == 0:
        return tensor
    rot = idx // 2
    flip = idx % 2 == 1
    out = torch.rot90(tensor, rot, dims=(-2, -1)) if rot else tensor
    if flip:
        out = torch.flip(out, dims=(-1,))
    return out.contiguous()


class EpisodeDataset(Dataset):
    """
    Loads episodes from `sim_dir` and exposes per-step n-step TD samples.

    Train/test split is at the episode level (seeded) to prevent leakage
    between correlated successive states within a trajectory.
    """

    def __init__(
        self,
        sim_dir: str | None = None,
        split: str = "train",
        test_fraction: float | None = None,
        split_seed: int | None = None,
        gamma: float | None = None,
        use_aug: bool | None = None,
        n_step: int | None = None,
    ) -> None:
        directory = sim_dir or param.SIMULATIONS_DIR
        test_frac = test_fraction if test_fraction is not None else param.TEST_SPLIT
        seed = split_seed if split_seed is not None else param.SPLIT_SEED
        gam = gamma if gamma is not None else param.GAMMA
        # Augmentation only on training data — leaks otherwise and inflates
        # apparent test-set size unhelpfully.
        augment = (use_aug if use_aug is not None else param.USE_DIHEDRAL_AUG) and split == "train"
        n = n_step if n_step is not None else param.TD_N_STEP

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

        # Unique state pool: one entry per (board, queue) appearing in any
        # episode.  Items reference these by index.  Memory cost is roughly
        # one tensor per timestep (vs. 8× pre-expansion in the old dataset).
        self._tensors: list[torch.Tensor] = []
        self._phis: list[float] = []
        # Each item:
        #   (idx_t, idx_next, rot_idx, n_step_reward_sum, bootstrap_flag)
        # bootstrap_flag is 1.0 when (t + n < T) and the episode didn't
        # terminate, else 0.0 (in which case idx_next can point anywhere — it
        # gets masked out — we still need a valid index so we set it to idx_t
        # for safety in collation).
        self._items: list[tuple[int, int, int, float, float]] = []

        gamma_powers = [gam ** k for k in range(n + 1)]

        for path in selected:
            episode = read_episode(path)
            steps = episode["steps"]
            T = len(steps)
            if T == 0:
                continue

            # Encode every step's state once; index = position in pool.
            base = len(self._tensors)
            for step in steps:
                board = Board.from_list(step["board"])
                queue: list[Piece] = [
                    PIECE_BY_ID[pid] for pid in step["queue"]
                    if pid in PIECE_BY_ID
                ]
                self._tensors.append(encode_state(board, queue))
                self._phis.append(board_potential(board.grid))

            rewards = [step["reward"] for step in steps]

            for t in range(T):
                horizon = min(n, T - t)
                n_step_sum = 0.0
                for k in range(horizon):
                    n_step_sum += gamma_powers[k] * rewards[t + k]
                # Bootstrap only if (t + n) is still inside the trajectory.
                # When t + n == T we fall back to the terminal-MC contribution
                # (V_target term is masked out via bootstrap_flag = 0).
                if t + n < T:
                    bootstrap = 1.0
                    idx_next = base + t + n
                else:
                    bootstrap = 0.0
                    idx_next = base + t  # placeholder; masked by bootstrap=0

                idx_t = base + t
                rot_variants = range(8) if augment else (0,)
                for rot in rot_variants:
                    self._items.append((idx_t, idx_next, rot, n_step_sum, bootstrap))

        # γ^n is constant per dataset; expose for the trainer.
        self.gamma_n: float = gamma_powers[n]
        self.n_step: int = n

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        idx_t, idx_next, rot, n_step_sum, bootstrap = self._items[idx]
        s_t = _apply_dihedral(self._tensors[idx_t], rot)
        # For non-bootstrap items s_next is unused; we still return a tensor
        # of matching shape (the same rotated s_t) so default_collate works
        # uniformly and the masked-out V_target contribution is well-defined.
        if bootstrap > 0.0:
            s_next = _apply_dihedral(self._tensors[idx_next], rot)
            phi_next = self._phis[idx_next]
        else:
            s_next = s_t
            phi_next = 0.0
        return (
            s_t,
            s_next,
            torch.tensor(n_step_sum, dtype=torch.float32),
            torch.tensor(self._phis[idx_t], dtype=torch.float32),
            torch.tensor(phi_next, dtype=torch.float32),
            torch.tensor(bootstrap, dtype=torch.float32),
        )
