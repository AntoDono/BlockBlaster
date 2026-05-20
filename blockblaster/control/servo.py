"""Closed-loop visual-servo placement (single-file rewrite).

Detection
=========
Colour- and palette-invariant.  Cache the grayscale board crop just
before DOWN as a baseline, then per frame compute
``cv2.absdiff(current_gray, baseline_gray)``, threshold to a binary
"this pixel moved" mask (morph-closed), and run
``cv2.matchTemplate`` of the known piece silhouette against that
motion mask to get a rigid initial pose.  The piece is the only thing
moving on the board, so the diff lights up exactly its rendered
footprint regardless of colour, translucency, or ghost-preview noise
from the game itself.  No tracker state between frames, no plant-gain
learning, no coarse open-loop jump, no fallbacks.

Per-frame measurement: 5 anchors
================================
Instead of one centroid we sample 5 anchors per frame and pair them
positionally with their target positions:

  - TL / TR / BL / BR: the extreme moving pixel in each corner
    direction of the corresponding extreme cell of the piece
    silhouette (e.g. for TL, the topmost-leftmost moving pixel in the
    top-row leftmost cell).  Anchored to crisp silhouette edges →
    robust to interior mass distribution.
  - C: centre-of-mass of motion in the most-central cell of the
    silhouette (elbow of an L, middle cell of a line, etc.).  Robust
    to corner occlusions which would bias corner anchors.

Anchors whose cell coverage falls below ``_CELL_MIN_COVERAGE`` are
dropped from the per-frame error.  The controller's error is the mean
of ``(target_i − measured_i)`` over the visible anchors, so partial
occlusions (board edges, score popups, animation flashes) don't bias
the average.

Gesture
=======
1. DOWN on the queue slot (centre nudged up by ``GRAB_Y_NUDGE_PX``),
   hold ``HOLD_MS`` so Block Blast registers the long-press grab.
2. Pre-lift diagonally to ``(board_centre_x, slot_y − INITIAL_LIFT_PX)``
   so wide pieces grabbed from edge slots don't hang off the board
   and lose detection.
3. Wait up to ``PRELIFT_CONFIRM_S`` for the matcher to confirm the
   piece is rendered (≥1 anchor + score ≥ ``LOCK_SCORE_MIN``).  Abort
   if it never confirms — better than blindly servoing a piece we
   can't see.
4. PD loop (below) until lock or budget.
5. ``PRE_LIFT_MS`` settle, then UP.

PD loop
=======
P scales with error (``err / GAIN``); D dampens by anticipating the
piece's motion (``DERIV_GAIN * derr / GAIN``).  If D flips the sign
of the P term it's nulled (let the piece coast one frame) — this
prevents the spiral overshoot we hit when D was too aggressive near
zero error.

The per-iter step ceiling is **distance-adaptive**:

  |err| ≥ FAR_ERR_PX   → cap = MAX_STEP_FAR_PX   (cover ground)
  |err| ≤ NEAR_ERR_PX  → cap = MAX_STEP_NEAR_PX  (fine alignment)
  in between           → linearly interpolated

Per-axis, so a piece aligned on x but far on y still gets a fast y
step without throwing x off.  ``MAX_STEP_FAR_PX`` is bounded by Block
Blast's drag follower lag — go too fast and the next-frame match
reads a stale position and the PD overshoots.

Release
=======
The UP commits the placement, so the gate is conservative:

  (tight_lock OR transit_lock) AND score ≥ LOCK_SCORE_MIN
                                AND paired ≥ LOCK_MIN_ANCHORS

  tight_lock:   |err_x| ≤ LOCK_TOL_PX AND |err_y| ≤ LOCK_TOL_PX
  transit_lock: each axis was inside LOCK_TOL_PX at some point this
                run AND is still within 2× tol now (catches a
                diagonal pass-through where the axes peak on
                different frames).

``LOCK_MIN_ANCHORS`` (default 2) lets edge placements release when
some corner anchors are off-board — the mean-of-visible error is
still accurate from as few as 2 anchors.

If the matcher loses the piece for ``MAX_NO_PIECE_FRAMES``
consecutive iters, abort.  If the ``MAX_LOOP_S`` budget elapses
without lock, lift in place rather than dragging back to the queue.

All tunables live in :mod:`blockblaster.config.params`.
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
    LOCK_MIN_ANCHORS,
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


# Per-anchor measurement: (anchor_idx, fx, fy, coverage_0_to_1).
# anchor_idx ∈ {0..4} indexes the list returned by _piece_anchors:
# 0=TL, 1=TR, 2=BL, 3=BR corners + 4=centroid of the most-central cell.
AnchorMeasurement = tuple[int, float, float, float]

# Anchor kinds: 4 corners + a centroid measurement of one cell.  The
# corner anchors are robust to interior mass distribution (anchored to
# silhouette edges); the centroid anchor is robust to corner noise
# (one corner cell going partially occluded biases corners; the
# central cell rarely is occluded).  Together they cross-check each
# other.
#   ("TL", "TR", "BL", "BR"): cx, cy ∈ {0, 1} edge of the chosen cell.
#   ("C",): centroid of that cell's motion mask.
AnchorKind = str  # 'TL' | 'TR' | 'BL' | 'BR' | 'C'
_ANCHOR_NAMES = ("TL", "TR", "BL", "BR", "C")

# Minimum fraction of moved pixels inside an anchor cell's window for
# that anchor to count as "visible".  Below this we drop it from the
# error average; lets the controller stay accurate when part of the
# piece is off-screen or occluded by score popups, etc.
_CELL_MIN_COVERAGE = 0.15


def _piece_anchors(
    piece: Piece,
) -> list[tuple[int, int, AnchorKind]]:
    """Return ``[(dr, dc, kind)]`` for the piece's 4 corners + 1 centroid.

    Corners are the extreme occupied cell of the silhouette in each
    direction (so non-rectangular pieces report corners that actually
    exist).  The centroid anchor uses the cell whose position is
    closest to the piece's geometric centre — for an O it's any of the
    four; for an L it's the elbow; for a 1×4 line it's a middle cell.
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


def _locate_piece(
    frame_bgr: np.ndarray,
    grid: CalibrationBox,
    piece: Piece,
    baseline_gray: np.ndarray,
) -> tuple[
    list[AnchorMeasurement], float,
    Optional[tuple[int, int]], np.ndarray,
]:
    """Return ``(anchors_measured, rigid_score, top_left_cell_rc, motion_mask)``.

    Pipeline:

    1. ``cv2.absdiff`` against the cached pre-drag baseline → threshold
       → morph-close = motion mask.
    2. ``cv2.matchTemplate`` of the piece's binary silhouette to get a
       rigid initial position.
    3. For each piece anchor (4 corners + 1 central cell centroid),
       inspect the corresponding cell's window in the motion mask:
       - Corner anchor: take the extreme moving pixel in that corner's
         direction (TL: topmost-leftmost; BR: bottommost-rightmost; …).
         Anchored to crisp silhouette edges, robust to interior mass.
       - Centroid anchor: take the centre-of-mass of motion pixels in
         the chosen central cell.  Robust to corner-cell occlusions
         which would bias corner anchors.
       The two anchor types cross-check each other.  Anchors whose
       cell coverage falls below :data:`_CELL_MIN_COVERAGE` are
       dropped so partial occlusions don't poison the error.

    ``anchors_measured`` is empty when the rigid match scores below
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

    # ── Per-anchor refinement ───────────────────────────────────────
    cell_area_px = cell_h * cell_w
    min_coverage_px = int(cell_area_px * _CELL_MIN_COVERAGE)
    anchors_measured: list[AnchorMeasurement] = []
    for idx, (dr, dc, kind) in enumerate(_piece_anchors(piece)):
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

    return anchors_measured, score, (tl_row, tl_col), search


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
        # Five target anchors of the piece silhouette in full-frame
        # pixels: 4 corner extremes + 1 centre-of-mass of the most-
        # central cell.  The controller's error is the mean of
        # (target_i − measured_i) over the anchors the matcher can
        # actually see this frame — partial occlusions stop biasing
        # the average.  Corners are anchored to silhouette edges
        # (low interior drift); the centroid is robust to corner
        # occlusions.  Together they cross-check each other.
        cell_w = cfg.grid.fw / BOARD_SIZE
        cell_h = cfg.grid.fh / BOARD_SIZE
        anchor_defs = _piece_anchors(suggestion.piece)
        target_anchors_xy: list[tuple[float, float]] = []
        for (dr, dc, kind) in anchor_defs:
            if kind == "C":
                ox, oy = 0.5, 0.5
            else:
                ox = 1.0 if kind in ("TR", "BR") else 0.0
                oy = 1.0 if kind in ("BL", "BR") else 0.0
            tcx = cfg.grid.fx + (suggestion.col + dc + ox) * cell_w
            tcy = cfg.grid.fy + (suggestion.row + dr + oy) * cell_h
            target_anchors_xy.append((tcx, tcy))
        target_cx_mean = sum(p[0] for p in target_anchors_xy) / len(target_anchors_xy)
        target_cy_mean = sum(p[1] for p in target_anchors_xy) / len(target_anchors_xy)

        if state is not None:
            state.servo_target_px = (int(target_cx_mean), int(target_cy_mean))
            state.servo_measured_px = None
            state.servo_target_cells = [(int(x), int(y)) for x, y in target_anchors_xy]
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
                anchors_measured, score, _, _ = _locate_piece(
                    frame, cfg.grid, suggestion.piece, baseline_gray,
                )
                if anchors_measured and score >= LOCK_SCORE_MIN:
                    confirmed = True
                    print(
                        f"[servo] pre-lift confirmed "
                        f"(score={score:.2f}, anchors={len(anchors_measured)}/5)"
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

                anchors_measured, score, tl_rc, motion = _locate_piece(
                    frame, cfg.grid, suggestion.piece, baseline_gray,
                )

                if not anchors_measured:
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

                # ── Per-anchor error: mean of (target_i − measured_i)
                # over only the anchors the matcher could actually see
                # this frame.  Paired by anchor index (TL/TR/BL/BR/C).
                measured_anchors_xy = [(c[1], c[2]) for c in anchors_measured]
                measured_by_idx = {c[0]: (c[1], c[2]) for c in anchors_measured}
                err_sum_x = 0.0
                err_sum_y = 0.0
                paired = 0
                for idx, (tx, ty) in enumerate(target_anchors_xy):
                    m = measured_by_idx.get(idx)
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
                meas_cx_mean = sum(p[0] for p in measured_anchors_xy) / len(measured_anchors_xy)
                meas_cy_mean = sum(p[1] for p in measured_anchors_xy) / len(measured_anchors_xy)

                if tl_rc is not None:
                    _publish(
                        (
                            tl_rc[1], tl_rc[0],
                            suggestion.piece.rows, suggestion.piece.cols,
                            score,
                        ),
                        mask=motion,
                        measured_cells=[(int(x), int(y)) for x, y in measured_anchors_xy],
                    )
                if state is not None:
                    state.servo_measured_px = (int(meas_cx_mean), int(meas_cy_mean))

                # Edge placements often only show 2-3 anchors (corner
                # cells off the board, score popup occlusion, etc.).
                # The mean-of-visible error is still accurate, so accept
                # the lock as long as at least LOCK_MIN_ANCHORS are
                # visible rather than requiring all 5.
                enough_anchors = paired >= LOCK_MIN_ANCHORS
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
                        and enough_anchors):
                    kind = "TIGHT" if tight_lock else "TRANSIT"
                    print(
                        f"[servo {iters}] LOCK[{kind}] err=({err_x:+d},{err_y:+d}) "
                        f"best=({best_err_x},{best_err_y}) "
                        f"score={score:.2f} anchors={paired}/5"
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
                    f"score={score:.2f} anchors={paired}/5 "
                    f"step=({dx:+d},{dy:+d}) finger={next_finger}"
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
