"""Per-frame piece detection: motion mask + template match + per-anchor refinement.

The pipeline is colour- and palette-invariant:

1. ``cv2.absdiff`` between the current grayscale board crop and a
   pre-DOWN baseline → threshold → morph-close = baseline motion
   mask.  Whatever the held piece looks like, the pixels it covers
   look *different* from what was there a moment ago.
2. Optional rolling mask (current vs. previous frame) for the
   caller's translation gate.  Silent on steady-state glow, bright
   wherever the piece physically moved this frame.
3. ``cv2.matchTemplate`` of the piece silhouette against the
   baseline mask = rigid initial pose.  When the caller seeds an
   ``expected_tl`` + ``search_radius_px``, the argmax is restricted
   to that window so faraway debris (glow blobs, score popups,
   cells that just brightened) is unreachable by construction.
4. Per-anchor refinement: 4 corner anchors (extreme moving pixel in
   each corner's direction within the corresponding extreme cell)
   + 1 centroid anchor (centre-of-mass of motion in the most-central
   cell).  Anchors whose cell coverage falls below
   :data:`CELL_MIN_COVERAGE` are dropped so partial occlusions don't
   poison the controller's error.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.assist.scanner import BOARD_SIZE
from blockblaster.config.params import (
    DIFF_THRESHOLD,
    MATCH_SCORE_MIN,
    MORPH_KERNEL_CELLS,
)
from blockblaster.game.pieces import Piece


# Per-anchor measurement: ``(anchor_idx, fx, fy, coverage_0_to_1)``.
# ``anchor_idx`` ∈ {0..4} indexes the list returned by
# :func:`piece_anchors`: 0=TL, 1=TR, 2=BL, 3=BR corners + 4=centroid
# of the most-central cell.
AnchorMeasurement = tuple[int, float, float, float]

# Anchor kinds: 4 corners + a centroid measurement of one cell.  The
# corner anchors are robust to interior mass distribution (anchored
# to silhouette edges); the centroid anchor is robust to corner noise
# (one corner cell going partially occluded biases corners; the
# central cell rarely is occluded).  Together they cross-check each
# other.
#   ("TL", "TR", "BL", "BR"): cx, cy ∈ {0, 1} edge of the chosen cell.
#   ("C",): centroid of that cell's motion mask.
AnchorKind = str  # 'TL' | 'TR' | 'BL' | 'BR' | 'C'
ANCHOR_NAMES = ("TL", "TR", "BL", "BR", "C")

# Minimum fraction of moved pixels inside an anchor cell's window
# for that anchor to count as "visible".  Below this we drop it from
# the error average; lets the controller stay accurate when part of
# the piece is off-screen or occluded by score popups, etc.
CELL_MIN_COVERAGE = 0.15


def board_gray(frame_bgr: np.ndarray, grid: CalibrationBox) -> np.ndarray:
    """Grayscale crop of the board area (baseline / per-frame input)."""
    crop = frame_bgr[grid.fy:grid.fy + grid.fh, grid.fx:grid.fx + grid.fw]
    if crop.size == 0:
        return np.zeros((max(1, grid.fh), max(1, grid.fw)), np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _motion_mask(
    current_gray: np.ndarray,
    baseline_gray: np.ndarray,
    threshold: int = DIFF_THRESHOLD,
    kernel_px: int = 0,
) -> np.ndarray:
    """Binary 'this pixel moved since baseline' mask.

    Frame-differencing is colour- and palette-invariant: whatever the
    held piece looks like, the pixels it covers on the board look
    *different* from what was there a moment ago.  When ``kernel_px``
    is supplied, a morph-close of that size fills the small gaps where
    the piece's rendered colour happens to land near the baseline
    brightness so the template correlates against a coherent blob,
    not a hollow outline.  Kernel size is cell-relative; the caller
    derives it from the calibrated grid.
    """
    if current_gray.shape != baseline_gray.shape:
        return np.zeros_like(current_gray)
    diff = cv2.absdiff(current_gray, baseline_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if kernel_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_px, kernel_px),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _make_template(piece: Piece, cell_h: int, cell_w: int) -> np.ndarray:
    """Binary silhouette of *piece* at the board's per-cell pixel scale."""
    h = piece.rows * cell_h
    w = piece.cols * cell_w
    tpl = np.zeros((h, w), np.uint8)
    for (dr, dc) in piece.cells:
        tpl[dr * cell_h:(dr + 1) * cell_h,
            dc * cell_w:(dc + 1) * cell_w] = 255
    return tpl


def piece_anchors(
    piece: Piece,
) -> list[tuple[int, int, AnchorKind]]:
    """Return ``[(dr, dc, kind)]`` for the piece's 4 corners + 1 centroid.

    Corners are the extreme occupied cell of the silhouette in each
    direction (so non-rectangular pieces report corners that actually
    exist).  The centroid anchor uses the cell whose position is
    closest to the piece's geometric centre — for an O it's any of
    the four; for an L it's the elbow; for a 1×4 line it's a middle
    cell.
    """
    top_dr = min(dr for dr, _ in piece.cells)
    bot_dr = max(dr for dr, _ in piece.cells)
    top_dcs = [dc for dr, dc in piece.cells if dr == top_dr]
    bot_dcs = [dc for dr, dc in piece.cells if dr == bot_dr]

    # Geometric centre of the piece in (dr, dc) space.
    cdr = sum(dr for dr, _ in piece.cells) / len(piece.cells)
    cdc = sum(dc for _, dc in piece.cells) / len(piece.cells)
    centre_dr, centre_dc = min(
        piece.cells,
        key=lambda c: (c[0] - cdr) ** 2 + (c[1] - cdc) ** 2,
    )

    return [
        (top_dr, min(top_dcs), "TL"),
        (top_dr, max(top_dcs), "TR"),
        (bot_dr, min(bot_dcs), "BL"),
        (bot_dr, max(bot_dcs), "BR"),
        (centre_dr, centre_dc, "C"),
    ]


def locate_piece(
    frame_bgr: np.ndarray,
    grid: CalibrationBox,
    piece: Piece,
    baseline_gray: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
    expected_tl_xy: Optional[tuple[int, int]] = None,
    search_radius_px: int = 0,
) -> tuple[
    list[AnchorMeasurement], float,
    Optional[tuple[int, int]], np.ndarray,
    Optional[np.ndarray], np.ndarray,
]:
    """Return ``(anchors, score, tl_rc, baseline_mask, rolling_mask, cur_gray)``.

    Pipeline:

    1. ``cv2.absdiff`` against the cached pre-drag baseline → threshold
       → morph-close = baseline motion mask.  This is what drives the
       template match (palette-invariant 'where is the piece sitting').
    2. When ``prev_gray`` is provided, also ``cv2.absdiff`` against the
       previous frame → ``DIFF_THRESHOLD`` → morph-close =
       rolling motion mask.  Silent on steady-state glow because glow
       cells stop *changing* once lit; bright wherever the piece
       physically moved since last frame.  Not used here for matching
       — the caller uses it as a translation gate on the detection.
    3. ``cv2.matchTemplate`` of the piece's binary silhouette against
       the baseline mask to get a rigid initial position.  When
       ``expected_tl_xy`` + ``search_radius_px`` are supplied, the
       argmax is restricted to a window of that half-extent around the
       expected top-left, so faraway debris (glow blobs, score popups,
       static cells that brightened) is unreachable by construction.
       Defaults are full-frame, matching the legacy behaviour used by
       the pre-lift confirmation pass which has no seed yet.
    4. For each piece anchor (4 corners + 1 central cell centroid),
       inspect the corresponding cell's window in the baseline mask:
       - Corner anchor: take the extreme moving pixel in that corner's
         direction (TL: topmost-leftmost; BR: bottommost-rightmost; …).
         Anchored to crisp silhouette edges, robust to interior mass.
       - Centroid anchor: take the centre-of-mass of motion pixels in
         the chosen central cell.  Robust to corner-cell occlusions
         which would bias corner anchors.
       The two anchor types cross-check each other.  Anchors whose
       cell coverage falls below :data:`CELL_MIN_COVERAGE` are
       dropped so partial occlusions don't poison the error.

    ``anchors_measured`` is empty when the rigid match scores below
    :data:`MATCH_SCORE_MIN`.  Both masks are always returned (the
    rolling mask is ``None`` when ``prev_gray`` is ``None``) so the
    GUI debug view can render them.  ``cur_gray`` is returned so the
    caller can promote it to ``prev_gray`` for the next iteration.
    """
    cell_h = max(1, grid.fh // BOARD_SIZE)
    cell_w = max(1, grid.fw // BOARD_SIZE)
    # Morph-close kernel sized to a fraction of a cell so it just-works
    # across phone resolutions.  Forced odd so the structuring element
    # has a well-defined centre.
    kernel_px = max(1, int(round(MORPH_KERNEL_CELLS * (cell_w + cell_h) / 2)))
    if kernel_px % 2 == 0:
        kernel_px += 1

    current_gray = board_gray(frame_bgr, grid)
    search = _motion_mask(current_gray, baseline_gray, kernel_px=kernel_px)
    rolling: Optional[np.ndarray] = None
    if prev_gray is not None and prev_gray.shape == current_gray.shape:
        rolling = _motion_mask(current_gray, prev_gray, kernel_px=kernel_px)

    tpl = _make_template(piece, cell_h, cell_w)

    if (search.shape[0] < tpl.shape[0]
            or search.shape[1] < tpl.shape[1]):
        return [], 0.0, None, search, rolling, current_gray

    result = cv2.matchTemplate(search, tpl, cv2.TM_CCORR_NORMED)

    # Restrict matchTemplate's argmax to a window around the expected
    # next top-left when the caller provides a seed.  The result tensor
    # has shape (H - tpl_h + 1, W - tpl_w + 1) in search-image coords,
    # so we slice it, run minMaxLoc on the slice, and offset the
    # returned top-left back into full search-image coords.
    if expected_tl_xy is not None and search_radius_px > 0:
        ex, ey = int(expected_tl_xy[0]), int(expected_tl_xy[1])
        r_y0 = max(0, ey - search_radius_px)
        r_x0 = max(0, ex - search_radius_px)
        r_y1 = min(result.shape[0], ey + search_radius_px + 1)
        r_x1 = min(result.shape[1], ex + search_radius_px + 1)
        if r_y1 <= r_y0 or r_x1 <= r_x0:
            # Window pushed entirely off the result tensor — treat as
            # "piece not findable here this frame".  Caller bumps
            # no_piece and the abort path takes over from there.
            return [], 0.0, None, search, rolling, current_gray
        result_win = result[r_y0:r_y1, r_x0:r_x1]
        _, score, _, top_left_win = cv2.minMaxLoc(result_win)
        top_left = (top_left_win[0] + r_x0, top_left_win[1] + r_y0)
    else:
        _, score, _, top_left = cv2.minMaxLoc(result)

    score = float(score)

    tl_col = int(round(top_left[0] / cell_w))
    tl_row = int(round(top_left[1] / cell_h))

    if score < MATCH_SCORE_MIN:
        return [], score, (tl_row, tl_col), search, rolling, current_gray

    # ── Per-anchor refinement ───────────────────────────────────────
    cell_area_px = cell_h * cell_w
    min_coverage_px = int(cell_area_px * CELL_MIN_COVERAGE)
    anchors_measured: list[AnchorMeasurement] = []
    for idx, (dr, dc, kind) in enumerate(piece_anchors(piece)):
        cy0 = top_left[1] + dr * cell_h
        cx0 = top_left[0] + dc * cell_w
        window = search[cy0:cy0 + cell_h, cx0:cx0 + cell_w]
        if window.size == 0:
            continue
        coverage_px = int(np.count_nonzero(window))
        if coverage_px < min_coverage_px:
            continue
        if kind == "C":
            m = cv2.moments(window, binaryImage=True)
            if m["m00"] == 0:
                continue
            local_x = m["m10"] / m["m00"]
            local_y = m["m01"] / m["m00"]
        else:
            ys, xs = np.where(window > 0)
            if xs.size == 0:
                continue
            # Extreme moving pixel in this corner's direction.
            local_x = float(xs.max() if kind in ("TR", "BR") else xs.min())
            local_y = float(ys.max() if kind in ("BL", "BR") else ys.min())
        fx = float(grid.fx + cx0 + local_x)
        fy = float(grid.fy + cy0 + local_y)
        anchors_measured.append((
            idx, fx, fy, coverage_px / float(cell_area_px),
        ))

    return anchors_measured, score, (tl_row, tl_col), search, rolling, current_gray
