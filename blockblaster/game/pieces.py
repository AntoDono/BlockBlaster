"""Canonical Block Blast piece definitions and sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Piece:
    piece_id: int
    name: str
    cells: tuple[tuple[int, int], ...]  # (row, col) offsets, top-left anchored

    @property
    def rows(self) -> int:
        return max(r for r, _ in self.cells) + 1

    @property
    def cols(self) -> int:
        return max(c for _, c in self.cells) + 1

    def to_grid(self) -> np.ndarray:
        """Return boolean 2-D grid of shape (rows, cols)."""
        grid = np.zeros((self.rows, self.cols), dtype=np.bool_)
        for r, c in self.cells:
            grid[r, c] = True
        return grid


def _piece(pid: int, name: str, rows: list[str]) -> Piece:
    cells: list[tuple[int, int]] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "X":
                cells.append((r, c))
    return Piece(piece_id=pid, name=name, cells=tuple(cells))


PIECES: list[Piece] = [
    # ── Single ──
    _piece(0,  "1x1",    ["X"]),

    # ── Horizontal bars ──
    _piece(1,  "1x2",    ["XX"]),
    _piece(2,  "1x3",    ["XXX"]),
    _piece(3,  "1x4",    ["XXXX"]),
    _piece(4,  "1x5",    ["XXXXX"]),

    # ── Vertical bars ──
    _piece(5,  "2x1",    ["X", "X"]),
    _piece(6,  "3x1",    ["X", "X", "X"]),
    _piece(7,  "4x1",    ["X", "X", "X", "X"]),
    _piece(8,  "5x1",    ["X", "X", "X", "X", "X"]),

    # ── Squares ──
    _piece(9,  "2x2",    ["XX", "XX"]),
    _piece(10, "3x3",    ["XXX", "XXX", "XXX"]),

    # ── Rectangles ──
    _piece(36, "2x3",    ["XXX", "XXX"]),               # 2 rows × 3 cols
    _piece(37, "3x2",    ["XX", "XX", "XX"]),           # 3 rows × 2 cols

    # ── L-shapes (2-cell leg) ──
    _piece(11, "L_2_TR", ["XX", "X."]),   # top-right foot
    _piece(12, "L_2_TL", ["XX", ".X"]),   # top-left foot
    _piece(13, "L_2_BR", ["X.", "XX"]),   # bottom-right foot
    _piece(14, "L_2_BL", [".X", "XX"]),   # bottom-left foot

    # ── L-shapes (3-cell leg + 1-cell foot, 4 cells total) ──
    _piece(15, "L_3_TR", ["XXX", "X.."]),
    _piece(16, "L_3_TL", ["XXX", "..X"]),
    _piece(17, "L_3_BR", ["X..", "XXX"]),
    _piece(18, "L_3_BL", ["..X", "XXX"]),
    _piece(19, "L_3_VR", ["X.", "X.", "XX"]),
    _piece(20, "L_3_VL", [".X", ".X", "XX"]),
    _piece(21, "L_3_VBR",["XX", "X.", "X."]),
    _piece(22, "L_3_VBL",["XX", ".X", ".X"]),

    # ── S / Z ──
    _piece(23, "S_H",    [".XX", "XX."]),
    _piece(24, "Z_H",    ["XX.", ".XX"]),
    _piece(25, "S_V",    ["X.", "XX", ".X"]),
    _piece(26, "Z_V",    [".X", "XX", "X."]),

    # ── T-shapes ──
    _piece(27, "T_U",    [".X.", "XXX"]),
    _piece(28, "T_D",    ["XXX", ".X."]),
    _piece(29, "T_L",    ["X.", "XX", "X."]),
    _piece(30, "T_R",    [".X", "XX", ".X"]),

    # ── Plus ──
    _piece(31, "PLUS",   [".X.", "XXX", ".X."]),

    # ── Diagonals (corner-touching cells, not edge-adjacent) ──
    _piece(32, "DIAG_2_BS", ["X.", ".X"]),               # "\" direction, length 2
    _piece(33, "DIAG_2_FS", [".X", "X."]),               # "/" direction, length 2
    _piece(34, "DIAG_3_BS", ["X..", ".X.", "..X"]),      # "\" direction, length 3
    _piece(35, "DIAG_3_FS", ["..X", ".X.", "X.."]),      # "/" direction, length 3

    # ── L-shapes (3-cell leg + 3-cell leg sharing a corner, 5 cells total) ──
    # All 4 rotations; suffix marks which corner of the 3×3 bbox the elbow is in.
    _piece(38, "L_5_TL", ["XXX", "X..", "X.."]),         # elbow top-left
    _piece(39, "L_5_TR", ["XXX", "..X", "..X"]),         # elbow top-right
    _piece(40, "L_5_BL", ["X..", "X..", "XXX"]),         # elbow bottom-left
    _piece(41, "L_5_BR", ["..X", "..X", "XXX"]),         # elbow bottom-right
]

PIECE_BY_ID: dict[int, Piece] = {p.piece_id: p for p in PIECES}
NUM_PIECES: int = len(PIECES)


def sample_queue(queue_size: int, rng: random.Random | None = None) -> list[Piece]:
    """Sample `queue_size` pieces uniformly at random (with replacement)."""
    if rng is None:
        return random.choices(PIECES, k=queue_size)
    return rng.choices(PIECES, k=queue_size)
