"""Greedy one-step placement suggester backed by a trained ValueNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import param
from blockblaster.game.board import Board, legal_positions_grid
from blockblaster.game.pieces import Piece
from blockblaster.game.potential import board_potential
from blockblaster.model.encoder import encode_state
from blockblaster.model.value_net import ValueNet


@dataclass(frozen=True)
class Suggestion:
    slot: int
    row: int
    col: int
    piece: Piece


def _place_and_clear(
    grid: np.ndarray, piece: Piece, row: int, col: int
) -> np.ndarray:
    g = grid.copy()
    for dr, dc in piece.cells:
        g[row + dr, col + dc] = 1
    full_rows = np.where(g.all(axis=1))[0]
    full_cols = np.where(g.all(axis=0))[0]
    if len(full_rows) or len(full_cols):
        g[full_rows, :] = 0
        g[:, full_cols] = 0
    return g


class Advisor:
    """Loads a ValueNet checkpoint and produces a single greedy suggestion."""

    def __init__(self, model_path: str | Path = "model.pt") -> None:
        self.model_path = Path(model_path)
        self.net: Optional[ValueNet] = None
        self.last_error: Optional[str] = None
        self._cache_key: Optional[tuple] = None
        self._cache_value: Optional[Suggestion] = None
        self._load()

    def clear_cache(self) -> None:
        """Forget the last (board, queue) → suggestion result."""
        self._cache_key = None
        self._cache_value = None

    def _load(self) -> None:
        if not self.model_path.exists():
            self.last_error = f"model not found: {self.model_path}"
            return
        try:
            net = ValueNet()
            data = torch.load(
                self.model_path,
                map_location=param.DEVICE,
                weights_only=True,
            )
            if isinstance(data, dict) and "state_dict" in data:
                net.load_state_dict(data["state_dict"])
            else:
                net.load_state_dict(data)  # type: ignore[arg-type]
            net.to(param.DEVICE)
            net.eval()
            self.net = net
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.net = None
            self.last_error = f"model load failed: {exc!r}"

    def suggest(
        self,
        board_grid: np.ndarray,
        queue: list[Optional[Piece]],
    ) -> Optional[Suggestion]:
        """Return the best single (slot, row, col) by greedy ValueNet scoring."""
        if self.net is None or not queue:
            return None

        original_slots = [i for i, p in enumerate(queue) if p is not None]
        pieces = [queue[i] for i in original_slots]
        if not pieces:
            return None

        grid = board_grid.astype(np.int8, copy=False)
        cache_key = (
            grid.tobytes(),
            grid.shape,
            tuple((i, p.piece_id) for i, p in zip(original_slots, pieces)),
        )
        if cache_key == self._cache_key:
            return self._cache_value

        # Build all candidate (slot_idx, row, col, next_grid) tuples.
        candidates: list[tuple[int, int, int, np.ndarray]] = []
        for slot_idx, piece in enumerate(pieces):
            for r, c in legal_positions_grid(grid, piece):
                candidates.append((slot_idx, r, c, _place_and_clear(grid, piece, r, c)))

        if not candidates:
            self._cache_key = cache_key
            self._cache_value = None
            return None

        # Score V*(s') = V_F(s') + Phi(s') for each candidate.
        board_obj = Board()
        tensors = []
        phis = []
        for slot_idx, _, _, next_grid in candidates:
            board_obj.grid = next_grid
            remaining = [p for i, p in enumerate(pieces) if i != slot_idx]
            tensors.append(encode_state(board_obj, remaining))
            phis.append(board_potential(next_grid))

        net_device = next(self.net.parameters()).device
        try:
            batch = torch.stack(tensors, dim=0).to(net_device)
            values = self.net.predict(batch)
            phi_t = torch.tensor(phis, dtype=values.dtype, device=values.device)
            scores = (values + phi_t).tolist()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"scoring failed: {exc!r}"
            self._cache_key = cache_key
            self._cache_value = None
            return None

        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        slot_idx, r, c, _ = candidates[best]
        suggestion = Suggestion(
            slot=original_slots[slot_idx],
            row=r,
            col=c,
            piece=pieces[slot_idx],
        )
        self._cache_key = cache_key
        self._cache_value = suggestion
        return suggestion
