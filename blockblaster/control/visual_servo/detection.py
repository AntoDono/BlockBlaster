"""Held-piece detection from a single board scan.

Block Blast renders the dragged piece in one of two ways depending on
device / cell colour / valid-drop highlighting:

* **Solid** — looks identical to a placed block.  ``scan_board`` reads
  its cells as placed.
* **Translucent** — alpha-blended over the cell.  ``scan_board`` reads
  it as either ghost-band or (over a previously-placed cell) as *empty*
  because the alpha drops the cell below the placed-threshold value.

To handle both, detection is the **union of two signals** against the
pre-grab snapshot of placed cells:

1. *New appearances*: cells that scan as placed-or-ghost now but didn't
   before.  Catches the piece sitting on empty board area.
2. *Occlusions*: cells that *were* placed pre-grab and no longer scan
   as placed.  Catches the piece sitting on top of existing blocks
   (and the resulting "raw_scan dropped by N cells" trace).

If the recon panel renders the piece, this layer agrees with it.
"""

from __future__ import annotations

import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.assist.scanner import scan_board, scan_board_with_ghost
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

    Combines two detection signals so the result is correct regardless
    of how Block Blast is rendering the dragged piece:

    * **New appearances** ``= (current_placed ∪ current_ghost) − initial_placed``
      — cells that look filled / translucent *now* but didn't before.
      This catches the piece floating over empty board cells whether
      it's drawn solid or alpha-blended.
    * **Occlusions** ``= initial_placed − current_placed``
      — cells that *were* placed pre-grab and no longer scan as placed,
      i.e. the piece is rendered on top of an existing block and
      dragged the cell's measured V below the placed threshold.

    Returns the union.  Either set being non-empty proves the piece is
    somewhere on the calibrated board.
    """
    placed, ghost = scan_board_with_ghost(frame, grid_box)
    current_placed = {(int(r), int(c)) for r, c in zip(*placed.nonzero())}
    current_ghost  = {(int(r), int(c)) for r, c in zip(*ghost.nonzero())}

    new_appearances = (current_placed | current_ghost) - initial_placed
    occlusions      = initial_placed - current_placed
    return new_appearances | occlusions


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
