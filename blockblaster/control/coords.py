"""Pixel-coordinate helpers: calibration boxes → screen pixel centres.

All inputs are :class:`~blockblaster.assist.calibration.CalibrationBox`
instances (stored in *original frame pixels*).  Returned coordinates are
also in frame pixels so they can be passed directly to
:meth:`~blockblaster.control.device.Device.swipe`.

Anchor convention
-----------------
All piece-to-pixel functions use the **bottom-row horizontal centre** of the
piece as the anchor.  Block Blast renders the dragged piece above the finger
with its bottom row closest to the finger tip, so this anchor is the most
consistent reference point across piece shapes.  A 1×1 piece has its only
cell as both the top and bottom row, so the anchor equals the cell centre.
"""

from __future__ import annotations

from typing import Sequence

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.game.pieces import Piece


def slot_center_px(queue_box: CalibrationBox, slot: int) -> tuple[int, int]:
    """Centre of queue slot *slot* (0–2) in frame pixels."""
    slot_w = queue_box.fw / 3
    cx = int(queue_box.fx + (slot + 0.5) * slot_w)
    cy = int(queue_box.fy + queue_box.fh / 2)
    return cx, cy


def cell_center_px(grid_box: CalibrationBox, row: int, col: int) -> tuple[int, int]:
    """Centre of grid cell ``(row, col)`` in frame pixels."""
    cell_w = grid_box.fw / 8
    cell_h = grid_box.fh / 8
    cx = int(grid_box.fx + (col + 0.5) * cell_w)
    cy = int(grid_box.fy + (row + 0.5) * cell_h)
    return cx, cy


def _bottom_row_center(piece: Piece) -> tuple[float, float]:
    """Return the fractional (row, col) of the bottom-row-centre anchor
    inside the piece's own coordinate frame (0-indexed, top-left = 0,0).

    For a 1×3 horizontal bar the anchor is (0, 1.0).
    For a 3×1 vertical bar the anchor is (2, 0.0).
    For an L-shape with cells (0,0),(1,0),(1,1) the anchor row=1, col=0.5.
    """
    max_r = max(r for r, _ in piece.cells)
    bottom_cols = [c for r, c in piece.cells if r == max_r]
    cf = sum(bottom_cols) / len(bottom_cols)
    return float(max_r), cf


def piece_anchor_px(
    grid_box: CalibrationBox,
    piece: Piece,
    row: int,
    col: int,
) -> tuple[int, int]:
    """Bottom-row-centre anchor of *piece* placed at board position ``(row, col)``.

    ``row`` and ``col`` are the top-left offset of the piece in board
    coordinates (matching :attr:`~blockblaster.assist.advisor.Suggestion.row`
    and ``.col``).
    """
    cell_w = grid_box.fw / 8
    cell_h = grid_box.fh / 8
    dr, dc = _bottom_row_center(piece)
    cx = int(grid_box.fx + (col + dc + 0.5) * cell_w)
    cy = int(grid_box.fy + (row + dr + 0.5) * cell_h)
    return cx, cy


def piece_anchor_px_from_cells(
    grid_box: CalibrationBox,
    cells: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Bottom-row-centre anchor computed from a list of *board-space* cells.

    Used by the calibrator to find the anchor of an observed landing:
    ``cells`` is the list of ``(row, col)`` pairs that were newly filled after
    a drag, in board coordinates.  The same bottom-row-centre rule is applied
    so the calibration samples are consistent with the live anchor used during
    auto-play.
    """
    max_r   = max(r for r, _ in cells)
    btm_cs  = [c for r, c in cells if r == max_r]
    cf      = sum(btm_cs) / len(btm_cs)
    cell_w  = grid_box.fw / 8
    cell_h  = grid_box.fh / 8
    cx = int(grid_box.fx + (cf + 0.5) * cell_w)
    cy = int(grid_box.fy + (max_r + 0.5) * cell_h)
    return cx, cy
