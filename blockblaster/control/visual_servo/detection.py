"""Held-piece detection from a single board scan.

The visual-servo placer uses **one** detection signal: the same
``scan_board`` output the recon panel renders.  Block Blast draws the
dragged piece on top of the board, so as the player (or our automated
finger) moves it, ``scan_board`` reads its cells as filled.  Subtract
the placed cells we saw *before* the grab and what's left is the live
position of the held piece.

No HSV ghost band, no "valid landing" preview detection, no separate
classifier.  If the GUI's recon panel says the piece is on the target
cell, the servo agrees by definition.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.assist.scanner import scan_board
from blockblaster.control.coords import piece_anchor_px_from_cells
from blockblaster.control.visual_servo.tunables import LOCK_TOLERANCE_PX


def snapshot_initial_placed(
    frame: Optional[np.ndarray],
    grid_box: CalibrationBox,
) -> set[tuple[int, int]]:
    """Snapshot the set of *truly* placed board cells before the grab.

    Returned as a ``{(row, col)}`` set, suitable for set-subtraction
    against per-iter scans inside the loop.  Returns an empty set on any
    error so a bad pre-frame can't crash the servo before it starts —
    the worst case is the loop sees a few extra "phantom" cells in the
    detected piece, which won't match expected and will be filtered by
    the cell-count check in :func:`is_locked`.
    """
    if frame is None:
        return set()
    try:
        placed = scan_board(frame, grid_box)
        return {(int(r), int(c)) for r, c in zip(*placed.nonzero())}
    except Exception:
        return set()


def detect_piece_cells(
    frame: np.ndarray,
    grid_box: CalibrationBox,
    initial_placed: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return the cells currently occupied by the held piece.

    Computed as ``scan_board(frame) − initial_placed`` — the difference
    isolates the moving piece from the permanently-placed blocks.
    """
    placed = scan_board(frame, grid_box)
    all_filled = {(int(r), int(c)) for r, c in zip(*placed.nonzero())}
    return all_filled - initial_placed


def is_locked(
    piece_cells:    set[tuple[int, int]],
    expected_cells: set[tuple[int, int]],
    piece_anchor:   tuple[int, int],
    target_anchor:  tuple[int, int],
) -> tuple[bool, str]:
    """Check whether the held piece is on the suggestion's target.

    Two ways to lock, in order of preference:

    1. **Exact match** — ``piece_cells == expected_cells``.  Ideal; only
       fires when the scanner sees exactly the right cells filled.
    2. **Tolerant match** — same cell *count* and the piece anchor is
       within ``LOCK_TOLERANCE_PX`` of the target on both axes.  This
       covers the case where the scanner flickers ±1 cell at a
       cell-boundary transition on a visually-correct placement; without
       it, the loop could spin forever on a perfect drop.

    Returns ``(is_locked, reason)``; ``reason`` is the string we'll log
    on lift.
    """
    if piece_cells == expected_cells:
        return True, "locked on piece"

    if len(piece_cells) == len(expected_cells):
        adx = target_anchor[0] - piece_anchor[0]
        ady = target_anchor[1] - piece_anchor[1]
        if abs(adx) <= LOCK_TOLERANCE_PX and abs(ady) <= LOCK_TOLERANCE_PX:
            return True, "locked (tolerant)"

    return False, ""


def piece_anchor(
    grid_box:    CalibrationBox,
    piece_cells: set[tuple[int, int]],
) -> tuple[int, int]:
    """Compute the held piece's anchor (frame pixels) from its cells."""
    return piece_anchor_px_from_cells(grid_box, list(piece_cells))
