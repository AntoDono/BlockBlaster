"""Potential function Phi(s) for reward shaping.

Following Ng, Harada, Russell (1999), we use potential-based shaping:

    F(s, s') = gamma * Phi(s') - Phi(s)

For Monte Carlo training this telescopes so the shaped return at time t is

    G_t^F = G_t - Phi(s_t)        (with Phi(terminal) := 0)

i.e. the only change to the training target is subtracting Phi at the *start*
state.  This preserves the original optimal policy *provided* that at action
selection time we add Phi(s') back to the network's prediction:

    a* = argmax_a [ V_F(s') + Phi(s') ]

Phi combines three complementary terms:

  1. Row/column fill (quadratic): rewards states with rows / columns that are
     close to full, giving the agent a dense gradient toward line-clear setups
     even before it has ever experienced one.

  2. Transition penalty (subtracted): counts adjacent cell pairs that flip
     between filled and empty, summed over all rows and all columns.
     Lower = better.  A solid block [1,1,1,1,0,0,0,0] has 1 transition per
     row; a checkerboard [1,0,1,0,1,0,1,0] has 7.  Penalises fragmented,
     interleaved boards where pieces are unlikely to complete lines.

  3. Piece fittability: for each of the 32 piece types, contributes
     |p| * num_legal_placements(p, board).  A board where the 3x3 square (or
     any large piece) has no legal placement contributes 0 for that piece.
     Direct signal that the agent is running out of room for certain pieces.

    Phi(s) = POTENTIAL_COEFF    * ( Σ_rows row_fill² + Σ_cols col_fill² )
           - TRANSITIONS_COEFF  * ( Σ_rows transitions(row) + Σ_cols transitions(col) )
           + FITTABILITY_COEFF  * Σ_{p ∈ PIECES} |p| * num_legal_placements(p, board)
"""

from __future__ import annotations

import numpy as np

import param
from blockblaster.game.pieces import PIECES, Piece


def _transitions(grid: np.ndarray) -> int:
    """Count internal filled<->empty transitions along rows and columns.

    For each row, count adjacent pairs that differ (0->1 or 1->0).
    Do the same for each column.  Boundaries are NOT counted.

    Examples on one row of length 8:
      [1, 1, 1, 1, 0, 0, 0, 0]  ->  1 transition
      [1, 0, 1, 0, 1, 0, 1, 0]  ->  7 transitions
    """
    row_trans = int((grid[:, :-1] != grid[:, 1:]).sum())
    col_trans = int((grid[:-1, :] != grid[1:, :]).sum())
    return row_trans + col_trans


def _fit_count(grid: np.ndarray, piece: Piece) -> int:
    """Return the number of legal top-left placements for `piece` on `grid`.

    Uses numpy slice intersection: for each piece cell (dr, dc), mask out
    positions where that cell would land on an occupied square.  The AND of all
    cell masks gives the set of valid top-left (row, col) positions.
    """
    n = grid.shape[0]
    pr, pc = piece.rows, piece.cols
    if pr > n or pc > n:
        return 0
    valid = np.ones((n - pr + 1, n - pc + 1), dtype=bool)
    for dr, dc in piece.cells:
        valid &= grid[dr : n - pr + 1 + dr, dc : n - pc + 1 + dc] == 0
    return int(valid.sum())


def _piece_fittability(grid: np.ndarray) -> float:
    """Return Σ_{p ∈ PIECES} |p| * num_legal_placements(p, grid)."""
    return float(sum(len(p.cells) * _fit_count(grid, p) for p in PIECES))


def board_potential(grid: np.ndarray) -> float:
    """Return Phi(grid).

    Three-term potential:

    1. Quadratic row/column fill: 7/8 is worth far more than 5/8.
       Coefficient scaled by `param.POTENTIAL_COEFF`.

    2. Transition penalty (subtracted): total filled<->empty flips across all
       rows and columns.  Lower transitions = more consolidated board.
       Coefficient scaled by `param.TRANSITIONS_COEFF`.

    3. Piece fittability: sum of |p| * num_legal_placements(p, board) over all
       32 piece types.  Boards where large pieces (e.g. the 3x3 square) have no
       legal placement are directly penalised.
       Coefficient scaled by `param.FITTABILITY_COEFF`.
    """
    g = grid.astype(np.float32, copy=False)
    rows = g.sum(axis=1)
    cols = g.sum(axis=0)
    fill        = param.POTENTIAL_COEFF   * float((rows * rows).sum() + (cols * cols).sum())
    transitions = param.TRANSITIONS_COEFF * _transitions(grid)
    fittability = param.FITTABILITY_COEFF * _piece_fittability(grid)
    return fill - transitions + fittability
