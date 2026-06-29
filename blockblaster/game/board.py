"""8x8 Block Blast board logic."""

from __future__ import annotations

import copy

import numpy as np

import param
from blockblaster.game.pieces import Piece


def legal_positions_grid(grid: np.ndarray, piece: Piece) -> list[tuple[int, int]]:
    """All ``(row, col)`` top-left positions where ``piece`` fits on ``grid``.

    Operates on a raw int/bool array (no :class:`Board` allocation) and uses a
    vectorised conjunction of shifted occupancy masks, which is the hot path
    for the beam-search policy and the live advisor.
    """
    n = grid.shape[0]
    pr, pc = piece.rows, piece.cols
    if pr > n or pc > n:
        return []
    valid = np.ones((n - pr + 1, n - pc + 1), dtype=bool)
    for dr, dc in piece.cells:
        valid &= grid[dr : n - pr + 1 + dr, dc : n - pc + 1 + dc] == 0
    rs, cs = np.where(valid)
    return list(zip(rs.tolist(), cs.tolist()))


class Board:
    """Mutable 8x8 grid.  Cells are 1 (occupied) or 0 (empty)."""

    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros(
            (param.BOARD_SIZE, param.BOARD_SIZE), dtype=np.int8
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def can_place(self, piece: Piece, row: int, col: int) -> bool:
        """Return True if `piece` fits when its top-left is at (row, col)."""
        for dr, dc in piece.cells:
            r, c = row + dr, col + dc
            if r < 0 or r >= param.BOARD_SIZE:
                return False
            if c < 0 or c >= param.BOARD_SIZE:
                return False
            if self.grid[r, c] != 0:
                return False
        return True

    def legal_placements(self, piece: Piece) -> list[tuple[int, int]]:
        """Return all (row, col) positions where `piece` can be placed."""
        return legal_positions_grid(self.grid, piece)

    def is_game_over(self, queue: list[Piece]) -> bool:
        """True when no piece in `queue` has any legal placement."""
        return all(len(self.legal_placements(p)) == 0 for p in queue)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def place(self, piece: Piece, row: int, col: int) -> int:
        """
        Place `piece` with top-left at (row, col).
        Returns the number of lines (rows + cols) cleared.
        Raises ValueError if the placement is illegal.
        """
        if not self.can_place(piece, row, col):
            raise ValueError(
                f"Cannot place piece '{piece.name}' at ({row}, {col})"
            )
        for dr, dc in piece.cells:
            self.grid[row + dr, col + dc] = 1
        return self._clear_lines()

    def _clear_lines(self) -> int:
        """Clear all full rows and columns simultaneously. Returns count cleared."""
        full_rows = [r for r in range(param.BOARD_SIZE) if self.grid[r].all()]
        full_cols = [c for c in range(param.BOARD_SIZE) if self.grid[:, c].all()]
        for r in full_rows:
            self.grid[r, :] = 0
        for c in full_cols:
            self.grid[:, c] = 0
        return len(full_rows) + len(full_cols)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clone(self) -> "Board":
        b = Board()
        b.grid = self.grid.copy()
        return b

    def to_list(self) -> list[list[int]]:
        return self.grid.tolist()

    @classmethod
    def from_list(cls, data: list[list[int]]) -> "Board":
        b = cls()
        b.grid = np.array(data, dtype=np.int8)
        return b

    def __repr__(self) -> str:  # pragma: no cover
        rows = ["".join("█" if cell else "." for cell in row) for row in self.grid]
        return "\n".join(rows)
