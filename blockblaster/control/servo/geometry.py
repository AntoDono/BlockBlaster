"""Board geometry helpers for target/error computation."""

from __future__ import annotations

import numpy as np

from blockblaster.assist.vision.scanner import BOARD_SIZE
from blockblaster.control.servo.config import (
    BOUNDARY_TOL_PX,
    FINE_STEP_PX,
    INITIAL_LIFT_PX,
    MIN_INITIAL_LIFT_PX,
)
from blockblaster.control.servo.types import Bbox
from blockblaster.game.pieces import Piece


def five_points(bbox: Bbox) -> list[tuple[int, int]]:
    """Centre + 4 bbox corners — the reference points aligned by the servo."""
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    cx, cy = (x + x1) // 2, (y + y1) // 2
    return [(cx, cy), (x, y), (x1, y), (x, y1), (x1, y1)]


def footprint_filled(
    baseline: np.ndarray,
    current: np.ndarray,
    piece: Piece,
    row: int,
    col: int,
) -> bool:
    """True when every target cell reads filled now but was empty pre-grab."""
    for dr, dc in piece.cells:
        r, c = row + dr, col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if baseline[r, c] or not current[r, c]:
            return False
    return True


def initial_lift_px(target_bbox: Bbox, grid_bbox: Bbox) -> int:
    """Less upward lift when the target sits on the bottom rows near the tray."""
    tgt_cy = target_bbox[1] + target_bbox[3] / 2
    gy, gh = grid_bbox[1], grid_bbox[3]
    if gh <= 0:
        return INITIAL_LIFT_PX
    board_pos = (tgt_cy - gy) / gh
    if board_pos <= 0.5:
        return INITIAL_LIFT_PX
    t = min(1.0, (board_pos - 0.5) / 0.5)
    scale = 1.0 - 0.5 * t
    return max(MIN_INITIAL_LIFT_PX, int(INITIAL_LIFT_PX * scale))


def push_toward_board_center(
    measured_bbox: Bbox, grid_bbox: Bbox, max_step: int = FINE_STEP_PX,
) -> tuple[int, int]:
    """Nudge the finger so the held piece moves back toward the board center."""
    mx = measured_bbox[0] + measured_bbox[2] / 2
    my = measured_bbox[1] + measured_bbox[3] / 2
    gx, gy, gw, gh = grid_bbox
    cx, cy = gx + gw / 2, gy + gh / 2
    err_x = cx - mx
    err_y = cy - my
    mag = (err_x ** 2 + err_y ** 2) ** 0.5
    if mag < 1:
        return 0, 0
    scale = min(max_step, mag) / mag
    return int(err_x * scale), int(err_y * scale)


def boundary_override(
    dx: int, dy: int, measured_bbox: Bbox, grid_bbox: Bbox,
) -> tuple[int, int, bool]:
    """If any piece corner drifts off the board, push it firmly back inward."""
    gx, gy, gw, gh = grid_bbox
    gx1, gy1 = gx + gw, gy + gh
    tol = BOUNDARY_TOL_PX
    corners = five_points(measured_bbox)[1:]
    breached = False
    if any(px < gx - tol for px, _ in corners):
        dx, breached = FINE_STEP_PX, True
    elif any(px > gx1 + tol for px, _ in corners):
        dx, breached = -FINE_STEP_PX, True
    if any(py < gy - tol for _, py in corners):
        dy, breached = FINE_STEP_PX, True
    elif any(py > gy1 + tol for _, py in corners):
        dy, breached = -FINE_STEP_PX, True
    return dx, dy, breached


def unobserved_cells(
    empty_mask: np.ndarray, grid_bbox: Bbox, obs_origin: tuple[int, int],
    cell_w: float, cell_h: float,
) -> list[Bbox]:
    """Cells the focus mask excludes (mostly filled), in frame px (xywh)."""
    gx, gy, _, _ = grid_bbox
    ox, oy = obs_origin
    cells: list[Bbox] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            fx0 = gx + int(c * cell_w)
            fx1 = gx + int((c + 1) * cell_w)
            fy0 = gy + int(r * cell_h)
            fy1 = gy + int((r + 1) * cell_h)
            cell = empty_mask[fy0 - oy:fy1 - oy, fx0 - ox:fx1 - ox]
            if cell.size and float(np.count_nonzero(cell)) / cell.size < 0.5:
                cells.append((fx0, fy0, fx1 - fx0, fy1 - fy0))
    return cells
