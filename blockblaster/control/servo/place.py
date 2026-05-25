"""Top-level ``place()`` orchestration for the visual servo.

See the package docstring (:mod:`blockblaster.control.servo`) for the
full design rationale.  This module ties together:

- detection (:mod:`.detection`),
- PD step / step cap / recovery direction (:mod:`.control`),
- touch + frame I/O (:mod:`.io`).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.calibration import CalibrationConfig
from blockblaster.assist.scanner import BOARD_SIZE
from blockblaster.config.params import (
    FAR_ERR_CELLS,
    FRAME_TIMEOUT_S,
    GLOW_AREA_RATIO,
    GLOW_HOLD_S,
    GRAB_Y_NUDGE_PX,
    HOLD_MS,
    INITIAL_LIFT_PX,
    LOCK_MIN_ANCHORS,
    LOCK_SCORE_MIN,
    LOCK_TOL_PX,
    MAX_LOOP_S,
    MAX_NO_PIECE_FRAMES,
    NEAR_ERR_CELLS,
    OFF_BOARD_HOLD_S,
    PRE_LIFT_MS,
    PRELIFT_CONFIRM_S,
    RECOVERY_STEP_CELLS,
    ROLLING_GATE_MIN_RATIO,
    ROLLING_GATE_RELAX_FACTOR,
    SEARCH_RADIUS_CELLS,
    SETTLE_MS,
    STEP_FAR_CELLS,
    STEP_NEAR_CELLS,
)
from blockblaster.control.coords import slot_center_px
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy

from .control import pd_control, step_cap
from .detection import board_gray, locate_piece, piece_anchors
from .io import move_smooth, wait_frame

if TYPE_CHECKING:
    from blockblaster.assist.app_state import AppState


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
    # Sentinel lets callers explicitly publish ``None`` for the rolling
    # mask (e.g. the first iter, when there's no prev frame) without us
    # interpreting that as "leave the field alone".
    _unset = object()

    def _publish(
        detection,
        mask=None,
        measured_cells=None,
        target_cells=None,
        rolling_mask=_unset,
    ):
        if state is not None:
            state.servo_detection = detection
            if mask is not None:
                state.servo_debug_mask = mask
            if rolling_mask is not _unset:
                state.servo_debug_mask_rolling = rolling_mask
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

        # ── Cell-relative px values, derived once per placement ──────
        # Almost everything in the servo (max step size, error
        # thresholds, search radius, gate relax zone) is naturally
        # expressed as a multiple of one board cell.  We resolve them
        # to pixels here so the loop can use plain integers, and the
        # ratios in params.py stay phone-independent.
        cell_w_int = max(1, cfg.grid.fw // BOARD_SIZE)
        cell_h_int = max(1, cfg.grid.fh // BOARD_SIZE)
        cell_px    = (cell_w_int + cell_h_int) / 2
        max_step_far_px       = max(1, int(round(STEP_FAR_CELLS  * cell_px)))
        max_step_near_px      = max(1, int(round(STEP_NEAR_CELLS * cell_px)))
        far_err_px            = max(1, int(round(FAR_ERR_CELLS   * cell_px)))
        near_err_px           = max(1, int(round(NEAR_ERR_CELLS  * cell_px)))
        search_radius_px      = max(1, int(round(SEARCH_RADIUS_CELLS * cell_px)))
        recovery_step_px      = max(1, int(round(RECOVERY_STEP_CELLS * cell_px)))
        rolling_gate_relax_px = max(1, int(round(
            ROLLING_GATE_RELAX_FACTOR * LOCK_TOL_PX,
        )))

        def _cap(err_mag: int) -> int:
            """Bound :func:`step_cap` to this placement's resolved params."""
            return step_cap(
                err_mag, near_err_px, far_err_px,
                max_step_near_px, max_step_far_px,
            )

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
        anchor_defs = piece_anchors(suggestion.piece)
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

        # Which corner anchors *should* be visible at the target?  A
        # corner is expected visible iff its piece-cell lands inside
        # the board at the target.  Used by the off-board recovery:
        # if a corner that should be visible isn't, the piece has
        # drifted off-board on that side and we push it back.  Edge
        # placements legitimately lose corners (e.g. a 2x2 at (6,6)
        # has all 4 corners on-board; a 2x2 at (6,7) only has TL+BL
        # on-board), so the "expected" set is target-dependent.
        expected_corners_at_target: set[int] = set()
        for idx, (dr, dc, kind) in enumerate(anchor_defs):
            if kind == "C":
                continue
            cell_row = suggestion.row + dr
            cell_col = suggestion.col + dc
            if (0 <= cell_row < BOARD_SIZE
                    and 0 <= cell_col < BOARD_SIZE):
                expected_corners_at_target.add(idx)

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
        baseline_gray = board_gray(pre_frame, cfg.grid)

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
            board_cy = int(round(cfg.grid.fy + cfg.grid.fh / 2))
            next_finger = (board_cx, down_px[1] - INITIAL_LIFT_PX)
            move_smooth(session, to_dev, down_px, next_finger)
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
            # Seed prev_gray with the last confirmation frame's gray
            # crop so the very first PD-loop iter already has a rolling
            # diff to gate on, rather than wasting an iter as no-prev.
            prev_gray: Optional[np.ndarray] = None
            # Pre-lift confirm also seeds the PD loop's local-search
            # window: by the time we enter the loop we have one trusted
            # top-left in board-crop pixels.  Without this seed, the
            # first iter would have to fall back to a full-frame match
            # — exactly the failure mode the window is meant to avoid.
            last_trusted_tl_px: Optional[tuple[int, int]] = None
            while time.monotonic() < confirm_deadline:
                frame, fid = wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid
                anchors_measured, score, tl_rc, _, _, cur_gray = locate_piece(
                    frame, cfg.grid, suggestion.piece, baseline_gray,
                )
                prev_gray = cur_gray
                if anchors_measured and score >= LOCK_SCORE_MIN:
                    confirmed = True
                    if tl_rc is not None:
                        last_trusted_tl_px = (
                            tl_rc[1] * cell_w_int,
                            tl_rc[0] * cell_h_int,
                        )
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
            # Last PD step we issued, in board-crop pixels.  Combined
            # with `last_trusted_tl_px` it gives the expected next
            # top-left location, which we centre the local matchTemplate
            # window on so the matcher can't latch onto faraway debris.
            last_commanded_dx = 0
            last_commanded_dy = 0
            # Pre-clear glow tracker: when Block Blast previews a
            # row/column clear, the motion mask balloons way past the
            # piece's silhouette area.  We only commit if that condition
            # holds for GLOW_HOLD_S consecutive seconds, so a transient
            # flash (score popup, etc.) can't trigger an early release.
            glow_start_t: Optional[float] = None
            piece_area_cells = len(suggestion.piece.cells)
            piece_area_px = piece_area_cells * cell_h_int * cell_w_int
            # Off-board recovery state.  When the corners that *should*
            # be visible at the target stop reporting for OFF_BOARD_HOLD_S,
            # we override the PD step with a fixed inward nudge until
            # they re-appear.
            off_board_start_t: Optional[float] = None

            while time.monotonic() < deadline:
                iters += 1
                frame, fid = wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid

                # Centre the matchTemplate window on where we expect
                # the piece to be this frame: last trusted location
                # plus the PD step we just commanded.  Defaults to
                # full-frame when no seed exists (pre-lift seeding
                # should always provide one, but the fallback keeps
                # the matcher functional if something goes wrong).
                if last_trusted_tl_px is not None:
                    expected_tl = (
                        last_trusted_tl_px[0] + last_commanded_dx,
                        last_trusted_tl_px[1] + last_commanded_dy,
                    )
                    search_radius = search_radius_px
                else:
                    expected_tl = None
                    search_radius = 0

                anchors_measured, score, tl_rc, motion, rolling, cur_gray = (
                    locate_piece(
                        frame, cfg.grid, suggestion.piece, baseline_gray,
                        prev_gray=prev_gray,
                        expected_tl_xy=expected_tl,
                        search_radius_px=search_radius,
                    )
                )

                # ── Pre-clear glow detector ──────────────────────────
                # If the motion mask covers far more area than the
                # piece's own silhouette could account for, Block
                # Blast is almost certainly rendering a row/column
                # clear glow under the held piece — which means we're
                # on an optimal placement.  Require GLOW_HOLD_S of
                # sustained glow before committing so a one-frame
                # flash can't trigger us.
                mask_area_px = int(np.count_nonzero(motion))
                if mask_area_px > GLOW_AREA_RATIO * piece_area_px:
                    now = time.monotonic()
                    if glow_start_t is None:
                        glow_start_t = now
                    elif now - glow_start_t >= GLOW_HOLD_S:
                        print(
                            f"[servo {iters}] PRE-CLEAR GLOW sustained "
                            f"{now - glow_start_t:.2f}s "
                            f"(mask={mask_area_px}px vs piece={piece_area_px}px) "
                            f"→ release"
                        )
                        time.sleep(PRE_LIFT_MS / 1000)
                        session.up()
                        return True
                else:
                    glow_start_t = None

                if not anchors_measured:
                    _publish(
                        None, mask=motion, measured_cells=[],
                        rolling_mask=rolling,
                    )
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
                    prev_gray = cur_gray
                    continue
                # NOTE: `no_piece` is *not* reset here.  The matcher
                # returned anchors, but they still have to clear the
                # rolling-diff translation gate below before we trust
                # them.  Resetting too early lets a placement loop
                # forever oscillating no_piece between 0 and 1 (matcher
                # finds the same phantom blob every frame, gate kills
                # it every frame).  The reset moved to right after the
                # gate passes.

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
                    _publish(
                        None, mask=motion, measured_cells=[],
                        rolling_mask=rolling,
                    )
                    no_piece += 1
                    prev_gray = cur_gray
                    continue
                err_x = int(err_sum_x / paired)
                err_y = int(err_sum_y / paired)

                # ── Rolling-diff translation gate ───────────────────
                # The baseline mask keeps glow blobs lit forever (it
                # diffs against a frozen pre-DOWN snapshot).  The
                # rolling mask only lights pixels that *changed this
                # frame*, so it's silent on steady-state glow but
                # bright where the piece actually moved.  Require some
                # rolling-motion coverage inside the matched footprint
                # — if the matcher latched onto a stationary glow blob
                # there'll be none, and we drop the frame.
                #
                # Relax the gate near lock-in: a correctly placed piece
                # has effectively stopped moving frame-to-frame, so
                # rolling coverage drops to zero through no fault of
                # the detection.  Trust the baseline match there.
                if (rolling is not None and tl_rc is not None
                        and max(abs(err_x), abs(err_y)) > rolling_gate_relax_px):
                    p_rows = suggestion.piece.rows
                    p_cols = suggestion.piece.cols
                    cell_h_g = max(1, cfg.grid.fh // BOARD_SIZE)
                    cell_w_g = max(1, cfg.grid.fw // BOARD_SIZE)
                    ty = max(0, tl_rc[0] * cell_h_g)
                    tx = max(0, tl_rc[1] * cell_w_g)
                    fh_px = p_rows * cell_h_g
                    fw_px = p_cols * cell_w_g
                    window = rolling[ty:ty + fh_px, tx:tx + fw_px]
                    footprint_px = max(1, len(suggestion.piece.cells)
                                       * cell_h_g * cell_w_g)
                    cov_px = int(np.count_nonzero(window))
                    cov = cov_px / float(footprint_px)
                    if cov < ROLLING_GATE_MIN_RATIO:
                        _publish(
                            None, mask=motion, measured_cells=[],
                            rolling_mask=rolling,
                        )
                        if state is not None:
                            state.servo_measured_px = None
                        no_piece += 1
                        print(
                            f"[servo {iters}] gated by rolling "
                            f"cov={cov:.2f} (need {ROLLING_GATE_MIN_RATIO:.2f}) "
                            f"err=({err_x:+d},{err_y:+d}) score={score:.2f}"
                        )
                        if no_piece >= MAX_NO_PIECE_FRAMES:
                            time.sleep(PRE_LIFT_MS / 1000)
                            session.up()
                            return False
                        prev_gray = cur_gray
                        continue

                # Frame is fully trusted: anchors found AND rolling
                # gate cleared (or relaxed at lock-in).  Reset the
                # no_piece counter and update the local-search seed so
                # the next iter's matchTemplate window centres on where
                # the piece actually is now.
                no_piece = 0
                if tl_rc is not None:
                    last_trusted_tl_px = (
                        tl_rc[1] * cell_w_int,
                        tl_rc[0] * cell_h_int,
                    )

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
                        rolling_mask=rolling,
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

                # PD step (per-axis): P chases the error, D anticipates
                # the piece's motion.  Distance-adaptive step cap then
                # bounds the integer finger delta.
                ctrl_x, derr_x = pd_control(err_x, prev_err_x)
                ctrl_y, derr_y = pd_control(err_y, prev_err_y)
                cap_x = _cap(abs(err_x))
                cap_y = _cap(abs(err_y))
                dx = max(-cap_x, min(cap_x, int(ctrl_x)))
                dy = max(-cap_y, min(cap_y, int(ctrl_y)))

                # ── Off-board corner recovery ───────────────────────
                # Compare currently-visible corner anchors against the
                # set we'd expect to see at the target.  Missing
                # corners that *should* be visible mean the piece
                # drifted off-board on the side those corners live —
                # push it back.  Hold for OFF_BOARD_HOLD_S to filter
                # transient occlusions (score popups, etc.).
                # Overrides the PD step on the affected axis; the
                # PD's err signal is unreliable here anyway (matcher
                # has too little piece to lock onto).
                visible_corners = {
                    idx for (idx, _, _, _) in anchors_measured
                    if idx < 4
                }
                missing_corners = (
                    expected_corners_at_target - visible_corners
                )
                now_t = time.monotonic()
                if missing_corners:
                    if off_board_start_t is None:
                        off_board_start_t = now_t
                else:
                    off_board_start_t = None
                recovery_active = (
                    off_board_start_t is not None
                    and now_t - off_board_start_t >= OFF_BOARD_HOLD_S
                )
                if recovery_active:
                    # Push the finger toward the board centre.  Robust
                    # to whichever side the piece drifted off — no
                    # need to disentangle matchTemplate saturation
                    # effects (which can make corners on the
                    # *opposite* side from the off-board edge appear
                    # empty) to figure out the direction.  Step is
                    # clamped per-axis so we don't overshoot the
                    # centre on a small residual offset.
                    ddx = board_cx - finger[0]
                    ddy = board_cy - finger[1]
                    rec_dx = int(max(-recovery_step_px,
                                     min(recovery_step_px, ddx)))
                    rec_dy = int(max(-recovery_step_px,
                                     min(recovery_step_px, ddy)))
                    if rec_dx != 0:
                        dx = rec_dx
                        # Null prev_err on the overridden axis so the
                        # next iter's D term doesn't compute against
                        # the (stale, off-board) PD error.
                        prev_err_x = None
                    if rec_dy != 0:
                        dy = rec_dy
                        prev_err_y = None
                    print(
                        f"[servo {iters}] OFF-BOARD RECOVERY → centre "
                        f"missing={sorted(missing_corners)} "
                        f"held={now_t - off_board_start_t:.2f}s "
                        f"step=({dx:+d},{dy:+d}) "
                        f"finger={finger} centre=({board_cx},{board_cy})"
                    )

                next_finger = (finger[0] + dx, finger[1] + dy)
                if not recovery_active:
                    print(
                        f"[servo {iters}] err=({err_x:+d},{err_y:+d}) "
                        f"derr=({derr_x:+d},{derr_y:+d}) "
                        f"score={score:.2f} anchors={paired}/5 "
                        f"step=({dx:+d},{dy:+d}) finger={next_finger}"
                    )
                prev_err_x, prev_err_y = err_x, err_y
                move_smooth(session, to_dev, finger, next_finger)
                finger = next_finger
                time.sleep(SETTLE_MS / 1000)
                # Record the PD step in board-crop pixels so next iter
                # can centre its matchTemplate window on
                # `last_trusted_tl + (dx, dy)`.  Finger px and
                # board-crop px share a scale (we're not in device-
                # touch coords here), so the delta is direct.
                last_commanded_dx = dx
                last_commanded_dy = dy
                # Promote the current frame's gray crop to prev for
                # next iter's rolling diff.  Updated *after* the move
                # so the next frame's rolling mask captures the motion
                # we just commanded.
                prev_gray = cur_gray

            # Budget exceeded — lift in place rather than drag back to queue.
            print(f"[servo] budget exceeded after {iters} iters")
            time.sleep(PRE_LIFT_MS / 1000)
            session.up()
            return False
    finally:
        if state is not None:
            state.servo_detection = None
            state.servo_debug_mask = None
            state.servo_debug_mask_rolling = None
            state.servo_target_px = None
            state.servo_measured_px = None
            state.servo_target_cells = []
            state.servo_measured_cells = []
