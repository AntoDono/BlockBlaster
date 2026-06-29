"""Closed-loop placement: grab tray piece, servo to target, release on recon lock."""

from __future__ import annotations

import random
import time
from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.vision.scanner import BOARD_SIZE, scan_board
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy
from blockblaster.control.servo.config import (
    APPROACH_RADIUS_PX,
    AREA_GROW_HOLD_STEP_PX,
    AREA_GROW_RELEASE_RATIO,
    AREA_GROW_STREAK,
    AREA_SHRINK_PUSH_RATIO,
    AREA_SHRINK_STREAK,
    FINE_STEP_PX,
    FRAME_TIMEOUT_S,
    HOLD_MS,
    INITIAL_AREA_DELAY_S,
    INITIAL_LIFT_SETTLE_MS,
    INITIAL_LIFT_SUBSTEP_MS,
    INITIAL_LIFT_SUBSTEPS,
    MAX_LOOP_S,
    MAX_NO_PIECE_FRAMES,
    PRE_LIFT_MS,
    RECON_LOCK_FRAMES,
    ROI_MARGIN_PX,
    SETTLE_MS,
    START_NOISE_X_PX,
)
from blockblaster.control.servo.geometry import (
    boundary_override,
    five_points,
    footprint_filled,
    initial_lift_px,
    push_toward_board_center,
    unobserved_cells,
)
from blockblaster.control.servo.motion import (
    clamp_step, move_smooth, pd_step, wait_frame,
)
from blockblaster.control.servo.tracking import board_gray, locate_piece
from blockblaster.control.servo.types import Bbox, DebugSink, LogSink, ServoDebug


def place(
    *,
    device: Device,
    grid_bbox: Bbox,
    grab_px: tuple[int, int],
    suggestion: Suggestion,
    frame_w: int,
    frame_h: int,
    on_debug: Optional[DebugSink] = None,
    on_log: Optional[LogSink] = None,
) -> bool:
    """DOWN on the tray piece, servo it onto the suggested cells, UP.

    ``grid_bbox`` is the board region in frame pixels (analyzer ``board_bbox``)
    and ``grab_px`` is where to press to pick the piece up. ``on_debug`` (if
    given) receives a :class:`ServoDebug` each iteration and ``None`` on exit.
    ``on_log`` (if given) receives human-readable status lines (also printed).
    Returns ``True`` only after recon lock (footprint filled for
    ``RECON_LOCK_FRAMES`` consecutive frames).
    """
    def emit(msg: str) -> None:
        print(msg)
        if on_log is not None:
            on_log(msg)

    def publish(dbg: Optional[ServoDebug]) -> None:
        if on_debug is None:
            return
        try:
            on_debug(dbg)
        except Exception:  # noqa: BLE001 — debug must never break the servo
            pass

    def release_on_recon_lock(
        session,
        *,
        iters: int,
        recon_streak: int,
        finger: tuple[int, int],
        measured_bbox: Optional[Bbox] = None,
        score: float = 0.0,
        initial_area_px: int = 0,
        current_area_px: int = 0,
        note: str = "",
    ) -> bool:
        """Release only when recon confirms footprint fill; sole path to servo ok."""
        if recon_streak < RECON_LOCK_FRAMES:
            return False
        suffix = f" — {note}" if note else ""
        publish(ServoDebug(
            target_bbox=target_bbox,
            measured_bbox=measured_bbox,
            finger_px=finger,
            observe_bbox=grid_bbox,
            score=score,
            locked=True,
            initial_area_px=initial_area_px,
            current_area_px=current_area_px,
            status="BOARD LOCK — RELEASING",
        ))
        emit(
            f"[servo {iters}] BOARD LOCK (recon, {recon_streak} frames){suffix}",
        )
        time.sleep(PRE_LIFT_MS / 1000)
        session.up()
        return True

    serial = getattr(device, "_serial", None)
    if not serial:
        emit("[servo] device has no serial; cannot inject input")
        return False
    try:
        dev_w, dev_h = device.screen_size()
    except Exception as exc:
        emit(f"[servo] screen_size failed: {exc}")
        return False

    dragger = get_scrcpy(serial, dev_w, dev_h)
    if dragger is None:
        return False

    sx = dev_w / max(1, frame_w)
    sy = dev_h / max(1, frame_h)

    def to_dev(p: tuple[int, int]) -> tuple[int, int]:
        return (int(round(p[0] * sx)), int(round(p[1] * sy)))

    gx, gy, gw, gh = grid_bbox
    cell_w = gw / BOARD_SIZE
    cell_h = gh / BOARD_SIZE
    cell_w_i = max(1, int(round(cell_w)))
    cell_h_i = max(1, int(round(cell_h)))

    piece = suggestion.piece
    tgt_x = int(gx + suggestion.col * cell_w)
    tgt_y = int(gy + suggestion.row * cell_h)
    tgt_w = int(piece.cols * cell_w)
    tgt_h = int(piece.rows * cell_h)
    target_bbox: Bbox = (tgt_x, tgt_y, tgt_w, tgt_h)
    target_pts = five_points(target_bbox)

    # Detection runs over the entire frame so an edge target or piece isn't
    # clipped by the board; board-cell math stays on grid_bbox.
    obs_region: Bbox = (0, 0, frame_w, frame_h)

    pre_frame, _ = device.get_latest_with_id()
    if pre_frame is None:
        emit("[servo] no pre-grab frame available")
        return False
    baseline_gray = board_gray(pre_frame, obs_region)
    baseline_board = scan_board(pre_frame, grid_bbox)
    _, empty_mask = cv2.threshold(
        baseline_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Local focus window around the target (clamped to the frame); near the
    # target we search only here ∩ empty cells.
    roi_x = max(0, tgt_x - ROI_MARGIN_PX)
    roi_y = max(0, tgt_y - ROI_MARGIN_PX)
    roi_x1 = min(frame_w, tgt_x + tgt_w + ROI_MARGIN_PX)
    roi_y1 = min(frame_h, tgt_y + tgt_h + ROI_MARGIN_PX)
    roi_bbox: Bbox = (roi_x, roi_y, roi_x1 - roi_x, roi_y1 - roi_y)
    roi_mask = np.zeros_like(empty_mask)
    roi_mask[roi_y:roi_y1, roi_x:roi_x1] = 255
    focus_mask = cv2.bitwise_and(empty_mask, roi_mask)

    unobserved = unobserved_cells(
        empty_mask, grid_bbox, (0, 0), cell_w, cell_h,
    )

    try:
        with dragger.open_session() as session:
            session.down(*to_dev(grab_px))
            time.sleep(HOLD_MS / 1000)

            jitter = random.randint(-START_NOISE_X_PX, START_NOISE_X_PX)
            lift_px = initial_lift_px(target_bbox, grid_bbox)
            finger = (grab_px[0] + jitter, grab_px[1] - lift_px)
            move_smooth(
                session, to_dev, grab_px, finger,
                steps=INITIAL_LIFT_SUBSTEPS,
                substep_ms=INITIAL_LIFT_SUBSTEP_MS,
            )
            time.sleep(INITIAL_LIFT_SETTLE_MS / 1000)

            initial_area_px = 0
            grow_streak = 0
            shrink_streak = 0
            offboard_push = False
            offboard_push_logged = False
            area_grow_hold = False
            time.sleep(INITIAL_AREA_DELAY_S)
            _, last_fid = device.get_latest_with_id()
            for _ in range(40):
                cal_frame, cal_fid = wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if cal_frame is not None:
                    last_fid = cal_fid
                    _, cal_score, cal_area = locate_piece(
                        cal_frame, obs_region, cell_w_i, cell_h_i, piece,
                        baseline_gray, None,
                    )
                    if cal_area > 0:
                        initial_area_px = cal_area
                        emit(
                            f"[servo] initial piece area: {initial_area_px}px "
                            f"(score={cal_score:.2f})",
                        )
                        break
                time.sleep(0.01)
            if initial_area_px == 0:
                emit("[servo] initial piece area: not detected — area guard off")

            deadline = time.monotonic() + MAX_LOOP_S
            no_piece = 0
            recon_streak = 0
            iters = 0
            prev_err_x: Optional[int] = None
            prev_err_y: Optional[int] = None
            # Last commanded step; reused during no-piece frames so the
            # drag keeps moving toward the target instead of pausing
            # whenever the blob detector momentarily drops the piece.
            last_dx = 0
            last_dy = 0

            while time.monotonic() < deadline:
                iters += 1
                frame, fid = wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid

                current_board = scan_board(frame, grid_bbox)
                if footprint_filled(
                    baseline_board, current_board, piece,
                    suggestion.row, suggestion.col,
                ):
                    recon_streak += 1
                else:
                    recon_streak = 0

                if release_on_recon_lock(
                    session,
                    iters=iters,
                    recon_streak=recon_streak,
                    finger=finger,
                    initial_area_px=initial_area_px,
                ):
                    return True

                # Gate board-aware focus on the *piece* error (not the finger),
                # so render lift between finger and piece doesn't shift modes
                # prematurely. While traveling we keep the full mask so a piece
                # crossing filled cells isn't ignored.
                near_target = (
                    prev_err_x is not None
                    and (prev_err_x ** 2 + prev_err_y ** 2) <= APPROACH_RADIUS_PX ** 2
                )
                observe_bbox = roi_bbox if near_target else grid_bbox

                measured_bbox, score, current_area_px = locate_piece(
                    frame, obs_region, cell_w_i, cell_h_i, piece,
                    baseline_gray, focus_mask if near_target else None,
                )
                if measured_bbox is None:
                    publish(ServoDebug(
                        target_bbox=target_bbox, finger_px=finger, score=score,
                        observe_bbox=observe_bbox, board_aware=near_target,
                        unobserved_cells=unobserved if near_target else [],
                        initial_area_px=initial_area_px,
                        status="SEARCHING FOR PIECE…",
                    ))
                    if recon_streak == 0 and not area_grow_hold:
                        no_piece += 1
                        if no_piece >= MAX_NO_PIECE_FRAMES:
                            emit(f"[servo] lost piece after {iters} iters; aborting")
                            time.sleep(PRE_LIFT_MS / 1000)
                            session.up()
                            return False
                    # Keep the gesture alive — replay the last commanded
                    # step so the drag keeps moving toward the target
                    # while the detector recovers.
                    if last_dx or last_dy:
                        next_finger = (finger[0] + last_dx, finger[1] + last_dy)
                        move_smooth(session, to_dev, finger, next_finger)
                        finger = next_finger
                        time.sleep(SETTLE_MS / 1000)
                    continue
                no_piece = 0

                if initial_area_px > 0 and current_area_px > 0:
                    area_ratio = current_area_px / initial_area_px
                    if area_ratio >= AREA_GROW_RELEASE_RATIO:
                        grow_streak += 1
                        shrink_streak = 0
                        offboard_push = False
                        offboard_push_logged = False
                        if grow_streak == AREA_GROW_STREAK:
                            emit(
                                f"[servo] area grew {current_area_px}/"
                                f"{initial_area_px} ({area_ratio:.2f}×) — "
                                "row clear, hold (1px) awaiting recon",
                            )
                        area_grow_hold = grow_streak >= AREA_GROW_STREAK
                    elif area_ratio <= AREA_SHRINK_PUSH_RATIO:
                        shrink_streak += 1
                        grow_streak = 0
                        area_grow_hold = False
                        offboard_push = shrink_streak >= AREA_SHRINK_STREAK
                        if offboard_push and not offboard_push_logged:
                            offboard_push_logged = True
                            emit(
                                f"[servo] area shrank {current_area_px}/"
                                f"{initial_area_px} ({area_ratio:.2f}×) — "
                                "off board, pushing to center",
                            )
                    else:
                        grow_streak = 0
                        shrink_streak = 0
                        offboard_push = False
                        offboard_push_logged = False
                        area_grow_hold = False

                # 5-point correspondence: error is the mean of target_i − measured_i.
                measured_pts = five_points(measured_bbox)
                err_x = int(sum(t[0] - m[0] for t, m in zip(target_pts, measured_pts)) / 5)
                err_y = int(sum(t[1] - m[1] for t, m in zip(target_pts, measured_pts)) / 5)

                if offboard_push:
                    dx, dy = push_toward_board_center(
                        measured_bbox, grid_bbox, max_step=FINE_STEP_PX,
                    )
                    dx, dy, breached = boundary_override(
                        dx, dy, measured_bbox, grid_bbox,
                    )
                    status = "OFF BOARD — PUSH TO CENTER"
                else:
                    dx, dy = pd_step(err_x, err_y, prev_err_x, prev_err_y, near_target)
                    dx, dy, breached = boundary_override(
                        dx, dy, measured_bbox, grid_bbox,
                    )
                    if breached:
                        status = "BOUNDARY HIT — PUSHING BACK"
                    elif near_target:
                        status = "FOCUSED — FINE APPROACH"
                    else:
                        status = "TRAVELING"

                if area_grow_hold and not offboard_push:
                    dx, dy = clamp_step(dx, dy, AREA_GROW_HOLD_STEP_PX)
                    status = "ROW CLEAR — HOLD (1px)"

                next_finger = (finger[0] + dx, finger[1] + dy)
                publish(ServoDebug(
                    target_bbox=target_bbox, measured_bbox=measured_bbox,
                    target_pts=target_pts, measured_pts=measured_pts,
                    finger_px=finger, err_px=(err_x, err_y),
                    step_px=(dx, dy), score=score, locked=False,
                    observe_bbox=observe_bbox, board_aware=near_target,
                    unobserved_cells=unobserved if near_target else [],
                    initial_area_px=initial_area_px,
                    current_area_px=current_area_px,
                    status=status,
                ))
                prev_err_x, prev_err_y = err_x, err_y
                last_dx, last_dy = dx, dy
                move_smooth(session, to_dev, finger, next_finger)
                finger = next_finger
                time.sleep(SETTLE_MS / 1000)

            emit(f"[servo] budget exceeded after {iters} iters; lifting in place")
            time.sleep(PRE_LIFT_MS / 1000)
            session.up()
            return False
    finally:
        publish(None)
