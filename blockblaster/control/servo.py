"""Closed-loop visual-servo placement (single-file rewrite).

Detection is color- and palette-invariant: cache the grayscale board
crop just before DOWN as a baseline, then per frame compute
``cv2.absdiff(current_gray, baseline_gray)``, threshold to a binary
"this pixel moved" mask, and refine the localisation by running
``cv2.matchTemplate`` of the known piece silhouette against that
motion mask.  The piece is the only thing moving on the board, so the
diff lights up exactly its rendered footprint regardless of colour,
translucency, or ghost-preview noise from the game itself.

No tracker state between frames, no plant-gain learning, no coarse
open-loop jump, no fallbacks.  If the matcher loses the piece for
``MAX_NO_PIECE_FRAMES`` consecutive iters, abort.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.calibration import CalibrationBox, CalibrationConfig
from blockblaster.assist.scanner import BOARD_SIZE
from blockblaster.config.params import (
    DERIV_GAIN,
    DIFF_THRESHOLD,
    FRAME_TIMEOUT_S,
    GAIN,
    GRAB_Y_NUDGE_PX,
    HOLD_MS,
    INITIAL_LIFT_PX,
    LOCK_SCORE_MIN,
    LOCK_TOL_PX,
    MATCH_SCORE_MIN,
    FAR_ERR_PX,
    MAX_LOOP_S,
    MAX_NO_PIECE_FRAMES,
    MAX_STEP_FAR_PX,
    MAX_STEP_NEAR_PX,
    MORPH_KERNEL_PX,
    MOVE_SUBSTEP_MS,
    MOVE_SUBSTEPS,
    NEAR_ERR_PX,
    PRE_LIFT_MS,
    PRELIFT_CONFIRM_S,
    SETTLE_MS,
)
from blockblaster.control.coords import slot_center_px
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy
from blockblaster.game.pieces import Piece

if TYPE_CHECKING:
    from blockblaster.assist.app_state import AppState


# All tunables live in :mod:`blockblaster.config.params`.  Re-tune there.


# ── Detection ─────────────────────────────────────────────────────────────

def _board_gray(frame_bgr: np.ndarray, grid: CalibrationBox) -> np.ndarray:
    """Grayscale crop of the board area, used as the baseline / per-frame input."""
    crop = frame_bgr[grid.fy:grid.fy + grid.fh, grid.fx:grid.fx + grid.fw]
    if crop.size == 0:
        return np.zeros((max(1, grid.fh), max(1, grid.fw)), np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _motion_mask(
    current_gray: np.ndarray, baseline_gray: np.ndarray,
) -> np.ndarray:
    """Binary 'this pixel moved since baseline' mask.

    Frame-differencing is colour- and palette-invariant: whatever the
    held piece looks like, the pixels it covers on the board look
    *different* from what was there a moment ago.  A morph-close fills
    the small gaps where the piece's rendered colour happens to land
    near the baseline brightness so the template correlates against a
    coherent blob, not a hollow outline.
    """
    if current_gray.shape != baseline_gray.shape:
        return np.zeros_like(current_gray)
    diff = cv2.absdiff(current_gray, baseline_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if MORPH_KERNEL_PX > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX),
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


# Per-cell measurement: (cell_dr, cell_dc, fx, fy, coverage_0_to_1).
PerCellMeasurement = tuple[int, int, float, float, float]

# Minimum fraction of moved pixels inside a cell window for that cell
# to count as "visible".  Below this we drop the cell from the error
# average; lets the controller stay accurate when part of the piece is
# off-screen or occluded by score popups, etc.
_CELL_MIN_COVERAGE = 0.15


def _locate_piece(
    frame_bgr: np.ndarray,
    grid: CalibrationBox,
    piece: Piece,
    baseline_gray: np.ndarray,
) -> tuple[
    list[PerCellMeasurement], float,
    Optional[tuple[int, int]], np.ndarray,
]:
    """Return ``(cells_measured, rigid_score, top_left_cell_rc, motion_mask)``.

    Pipeline:

    1. ``cv2.absdiff`` against the cached pre-drag baseline → threshold
       → morph-close = motion mask.
    2. ``cv2.matchTemplate`` of the piece's binary silhouette to get a
       rigid initial position.
    3. For each ``(dr, dc)`` of the piece, run ``cv2.moments`` on the
       cell-sized window at the matched top-left + ``(dr*ch, dc*cw)``
       to get that cell's actual center-of-mass.  Drop cells whose
       motion coverage is below :data:`_CELL_MIN_COVERAGE` so partial
       occlusions don't bias the controller's error.

    ``cells_measured`` is empty when the rigid match scores below
    :data:`MATCH_SCORE_MIN`.  ``motion_mask`` is always returned for
    the GUI debug view.
    """
    current_gray = _board_gray(frame_bgr, grid)
    search = _motion_mask(current_gray, baseline_gray)

    cell_h = max(1, grid.fh // BOARD_SIZE)
    cell_w = max(1, grid.fw // BOARD_SIZE)
    tpl = _make_template(piece, cell_h, cell_w)

    if (search.shape[0] < tpl.shape[0]
            or search.shape[1] < tpl.shape[1]):
        return [], 0.0, None, search

    result = cv2.matchTemplate(search, tpl, cv2.TM_CCORR_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(result)
    score = float(score)

    tl_col = int(round(top_left[0] / cell_w))
    tl_row = int(round(top_left[1] / cell_h))

    if score < MATCH_SCORE_MIN:
        return [], score, (tl_row, tl_col), search

    # ── Per-cell refinement ─────────────────────────────────────────
    cell_area_px = cell_h * cell_w
    min_coverage_px = int(cell_area_px * _CELL_MIN_COVERAGE)
    cells_measured: list[PerCellMeasurement] = []
    for (dr, dc) in piece.cells:
        cy0 = top_left[1] + dr * cell_h
        cx0 = top_left[0] + dc * cell_w
        window = search[cy0:cy0 + cell_h, cx0:cx0 + cell_w]
        if window.size == 0:
            continue
        coverage_px = int(np.count_nonzero(window))
        if coverage_px < min_coverage_px:
            continue
        m = cv2.moments(window, binaryImage=True)
        if m["m00"] == 0:
            continue
        local_x = m["m10"] / m["m00"]
        local_y = m["m01"] / m["m00"]
        # Full-frame pixel coordinates of this cell's centre-of-mass.
        fx = float(grid.fx + cx0 + local_x)
        fy = float(grid.fy + cy0 + local_y)
        cells_measured.append((
            int(dr), int(dc), fx, fy, coverage_px / float(cell_area_px),
        ))

    return cells_measured, score, (tl_row, tl_col), search


# ── Motion helper ─────────────────────────────────────────────────────────

def _move_smooth(
    session, to_dev, start_xy: tuple[int, int], end_xy: tuple[int, int],
) -> None:
    """Interpolate a single move into ``MOVE_SUBSTEPS`` touch events.

    ``start_xy`` and ``end_xy`` are in frame pixels.  Spacing the
    intermediate moves by ``MOVE_SUBSTEP_MS`` lets Block Blast's drag
    follower render the piece continuously instead of jumping, which
    keeps the next-frame matcher reading accurate.
    """
    steps = max(1, MOVE_SUBSTEPS)
    for i in range(1, steps + 1):
        t = i / steps
        x = int(round(start_xy[0] + (end_xy[0] - start_xy[0]) * t))
        y = int(round(start_xy[1] + (end_xy[1] - start_xy[1]) * t))
        session.move(*to_dev((x, y)))
        if i < steps and MOVE_SUBSTEP_MS > 0:
            time.sleep(MOVE_SUBSTEP_MS / 1000)


# ── Frame helper ──────────────────────────────────────────────────────────

def _wait_frame(
    device: Device, last_fid: int, timeout_s: float,
) -> tuple[Optional[np.ndarray], int]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_fid:
            return frame, fid
        time.sleep(0.005)
    return None, last_fid


# ── Public entrypoint ─────────────────────────────────────────────────────

def place(
    *,
    device:     Device,
    cfg:        CalibrationConfig,
    suggestion: Suggestion,
    frame_w:    int,
    frame_h:    int,
    state:      Optional["AppState"] = None,
) -> bool:
    """DOWN on the queue slot, drag to the suggested cells, UP.

    Returns ``True`` on confident release, ``False`` if the held piece
    couldn't be tracked or the loop budget elapsed.  When ``state`` is
    provided, the latest detection is published on
    ``state.servo_detection`` for live GUI visualization.
    """
    def _publish(detection, mask=None, measured_cells=None, target_cells=None):
        if state is not None:
            state.servo_detection = detection
            if mask is not None:
                state.servo_debug_mask = mask
            if measured_cells is not None:
                state.servo_measured_cells = measured_cells
            if target_cells is not None:
                state.servo_target_cells = target_cells

    try:
        if cfg.grid is None or not cfg.grid.is_valid():
            return False
        if cfg.queue is None or not cfg.queue.is_valid():
            return False

        serial = getattr(device, "_serial", None)
        if not serial:
            return False
        try:
            dev_w, dev_h = device.screen_size()
        except Exception as exc:
            print(f"[servo] screen_size failed: {exc}")
            return False

        dragger = get_scrcpy(serial, dev_w, dev_h)
        if dragger is None:
            return False

        sx = dev_w / max(1, frame_w)
        sy = dev_h / max(1, frame_h)
        def to_dev(p: tuple[int, int]) -> tuple[int, int]:
            return (int(round(p[0] * sx)), int(round(p[1] * sy)))

        # ── Targets ───────────────────────────────────────────────────
        # Per-cell target centres: one (fx, fy) per piece cell, in
        # full-frame pixels.  The controller's error is the mean of
        # (target_i − measured_i) over the cells the matcher can
        # actually see this frame — partial occlusions stop biasing
        # the average.
        cell_w = cfg.grid.fw / BOARD_SIZE
        cell_h = cfg.grid.fh / BOARD_SIZE
        target_cells_xy: list[tuple[float, float]] = []
        for (dr, dc) in suggestion.piece.cells:
            tcx = cfg.grid.fx + (suggestion.col + dc + 0.5) * cell_w
            tcy = cfg.grid.fy + (suggestion.row + dr + 0.5) * cell_h
            target_cells_xy.append((tcx, tcy))
        target_cx_mean = sum(p[0] for p in target_cells_xy) / len(target_cells_xy)
        target_cy_mean = sum(p[1] for p in target_cells_xy) / len(target_cells_xy)

        if state is not None:
            state.servo_target_px = (int(target_cx_mean), int(target_cy_mean))
            state.servo_measured_px = None
            state.servo_target_cells = [(int(x), int(y)) for x, y in target_cells_xy]
            state.servo_measured_cells = []

        slot_cx, slot_cy = slot_center_px(cfg.queue, suggestion.slot)
        down_px = (slot_cx, slot_cy - GRAB_Y_NUDGE_PX)

        # ── Snapshot the pre-drag board ───────────────────────────────
        # The baseline grayscale crop is what the per-frame diff
        # subtracts against.  Captured *before* DOWN, while the board is
        # static, so the only thing that changes in subsequent frames
        # is the held piece's rendered footprint.
        pre_frame, _ = device.get_latest_with_id()
        if pre_frame is None:
            print("[servo] no pre-grab frame available")
            return False
        baseline_gray = _board_gray(pre_frame, cfg.grid)

        # ── Gesture ───────────────────────────────────────────────────
        with dragger.open_session() as session:
            session.down(*to_dev(down_px))
            time.sleep(HOLD_MS / 1000)

            # Pre-lift to the board's horizontal centre (not straight up
            # from the queue slot).  A long horizontal piece grabbed from
            # an edge queue slot would otherwise hang off the board and
            # never get fully rendered → the matcher can't find it.
            # Centering X keeps the piece's footprint on-board regardless
            # of which slot it came from.
            board_cx = int(round(cfg.grid.fx + cfg.grid.fw / 2))
            next_finger = (board_cx, down_px[1] - INITIAL_LIFT_PX)
            _move_smooth(session, to_dev, down_px, next_finger)
            finger = next_finger
            time.sleep(SETTLE_MS / 1000)

            # ── Wait for the piece to actually be rendered & detected at
            # the centered pre-lift position before letting the PD loop
            # take over.  Otherwise the first few iters either count as
            # `no_piece` (piece not yet drawn) or steer off a half-drawn
            # match while Block Blast's drag follower is still catching
            # up to the pre-lift move.  We give it up to PRELIFT_CONFIRM_S
            # seconds to produce one good detection; if it never does, we
            # abort the placement rather than blindly servoing.
            _, last_fid = device.get_latest_with_id()
            confirm_deadline = time.monotonic() + PRELIFT_CONFIRM_S
            confirmed = False
            while time.monotonic() < confirm_deadline:
                frame, fid = _wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid
                cells_measured, score, _, _ = _locate_piece(
                    frame, cfg.grid, suggestion.piece, baseline_gray,
                )
                if cells_measured and score >= LOCK_SCORE_MIN:
                    confirmed = True
                    print(
                        f"[servo] pre-lift confirmed "
                        f"(score={score:.2f}, cells={len(cells_measured)})"
                    )
                    break
            if not confirmed:
                print("[servo] pre-lift confirmation timeout; aborting")
                time.sleep(PRE_LIFT_MS / 1000)
                session.up()
                return False

            deadline = time.monotonic() + MAX_LOOP_S
            no_piece   = 0
            iters      = 0
            prev_err_x: Optional[int] = None
            prev_err_y: Optional[int] = None
            best_err_x = 10**9   # min |err_x| seen this run
            best_err_y = 10**9   # min |err_y| seen this run

            while time.monotonic() < deadline:
                iters += 1
                frame, fid = _wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid

                cells_measured, score, tl_rc, motion = _locate_piece(
                    frame, cfg.grid, suggestion.piece, baseline_gray,
                )

                if not cells_measured:
                    _publish(None, mask=motion, measured_cells=[])
                    if state is not None:
                        state.servo_measured_px = None
                    no_piece += 1
                    print(
                        f"[servo {iters}] no piece "
                        f"(score={score:.2f}, count={no_piece})"
                    )
                    if no_piece >= MAX_NO_PIECE_FRAMES:
                        time.sleep(PRE_LIFT_MS / 1000)
                        session.up()
                        return False
                    continue
                no_piece = 0

                # ── Per-cell error: mean of (target_i − measured_i)
                # over only the cells the matcher could actually see
                # this frame.
                measured_cells_xy = [(c[2], c[3]) for c in cells_measured]
                measured_set = {(c[0], c[1]): (c[2], c[3]) for c in cells_measured}
                err_sum_x = 0.0
                err_sum_y = 0.0
                paired = 0
                for (dr, dc), (tx, ty) in zip(suggestion.piece.cells, target_cells_xy):
                    m = measured_set.get((dr, dc))
                    if m is None:
                        continue
                    err_sum_x += tx - m[0]
                    err_sum_y += ty - m[1]
                    paired += 1
                if paired == 0:
                    _publish(None, mask=motion, measured_cells=[])
                    no_piece += 1
                    continue
                err_x = int(err_sum_x / paired)
                err_y = int(err_sum_y / paired)

                # Aggregate "where is the piece" for the headline dot.
                meas_cx_mean = sum(p[0] for p in measured_cells_xy) / len(measured_cells_xy)
                meas_cy_mean = sum(p[1] for p in measured_cells_xy) / len(measured_cells_xy)

                if tl_rc is not None:
                    _publish(
                        (
                            tl_rc[1], tl_rc[0],
                            suggestion.piece.rows, suggestion.piece.cols,
                            score,
                        ),
                        mask=motion,
                        measured_cells=[(int(x), int(y)) for x, y in measured_cells_xy],
                    )
                if state is not None:
                    state.servo_measured_px = (int(meas_cx_mean), int(meas_cy_mean))

                all_cells_visible = paired == len(suggestion.piece.cells)
                best_err_x = min(best_err_x, abs(err_x))
                best_err_y = min(best_err_y, abs(err_y))

                # Two release conditions:
                #   1. Tight lock: both axes inside LOCK_TOL_PX *right now*.
                #   2. Transit lock: each axis has been inside LOCK_TOL_PX at
                #      some point this run AND is still within 2x tol now.
                #      Catches a diagonal pass-through where the axes peak
                #      on different frames.
                tight_lock = (abs(err_x) <= LOCK_TOL_PX
                              and abs(err_y) <= LOCK_TOL_PX)
                transit_lock = (best_err_x <= LOCK_TOL_PX
                                and best_err_y <= LOCK_TOL_PX
                                and abs(err_x) <= 2 * LOCK_TOL_PX
                                and abs(err_y) <= 2 * LOCK_TOL_PX)
                if ((tight_lock or transit_lock)
                        and score >= LOCK_SCORE_MIN
                        and all_cells_visible):
                    kind = "TIGHT" if tight_lock else "TRANSIT"
                    print(
                        f"[servo {iters}] LOCK[{kind}] err=({err_x:+d},{err_y:+d}) "
                        f"best=({best_err_x},{best_err_y}) "
                        f"score={score:.2f} cells={paired}/"
                        f"{len(suggestion.piece.cells)}"
                    )
                    time.sleep(PRE_LIFT_MS / 1000)
                    session.up()
                    return True

                # PD step: P chases the error, D dampens by anticipating
                # the piece's motion.  When err is shrinking (piece is
                # already heading toward the target), `derr` has the
                # opposite sign of `err` and the step shrinks.
                #
                # Critical: clamp the D contribution so it can *reduce* the
                # P term but never flip its sign — otherwise an aggressive
                # D term will push the finger past the target when err is
                # already near zero, which manifests as the spiral overshoot
                # we saw in testing.
                derr_x = 0 if prev_err_x is None else err_x - prev_err_x
                derr_y = 0 if prev_err_y is None else err_y - prev_err_y
                p_x = err_x / GAIN
                p_y = err_y / GAIN
                d_x = DERIV_GAIN * derr_x / GAIN
                d_y = DERIV_GAIN * derr_y / GAIN
                ctrl_x = p_x + d_x
                ctrl_y = p_y + d_y
                # If D dragged the control past zero (sign flipped vs P),
                # null it out — let the piece coast for one frame instead.
                if (p_x >= 0) != (ctrl_x >= 0):
                    ctrl_x = 0.0
                if (p_y >= 0) != (ctrl_y >= 0):
                    ctrl_y = 0.0
                # Distance-adaptive step ceiling: big jumps when far,
                # small precise steps when close.  Per-axis so a piece
                # aligned on x but far on y still gets a fast y step
                # without throwing x off.
                def _step_cap(err_mag: int) -> int:
                    if err_mag >= FAR_ERR_PX:
                        return MAX_STEP_FAR_PX
                    if err_mag <= NEAR_ERR_PX:
                        return MAX_STEP_NEAR_PX
                    span = max(1, FAR_ERR_PX - NEAR_ERR_PX)
                    t = (err_mag - NEAR_ERR_PX) / span
                    return int(round(
                        MAX_STEP_NEAR_PX
                        + t * (MAX_STEP_FAR_PX - MAX_STEP_NEAR_PX)
                    ))
                cap_x = _step_cap(abs(err_x))
                cap_y = _step_cap(abs(err_y))
                dx = max(-cap_x, min(cap_x, int(ctrl_x)))
                dy = max(-cap_y, min(cap_y, int(ctrl_y)))
                next_finger = (finger[0] + dx, finger[1] + dy)
                print(
                    f"[servo {iters}] err=({err_x:+d},{err_y:+d}) "
                    f"derr=({derr_x:+d},{derr_y:+d}) "
                    f"score={score:.2f} cells={paired}/"
                    f"{len(suggestion.piece.cells)} step=({dx:+d},{dy:+d}) "
                    f"finger={next_finger}"
                )
                prev_err_x, prev_err_y = err_x, err_y
                _move_smooth(session, to_dev, finger, next_finger)
                finger = next_finger
                time.sleep(SETTLE_MS / 1000)

            # Budget exceeded — lift in place rather than drag back to queue.
            print(f"[servo] budget exceeded after {iters} iters")
            time.sleep(PRE_LIFT_MS / 1000)
            session.up()
            return False
    finally:
        if state is not None:
            state.servo_detection = None
            state.servo_debug_mask = None
            state.servo_target_px = None
            state.servo_measured_px = None
            state.servo_target_cells = []
            state.servo_measured_cells = []
