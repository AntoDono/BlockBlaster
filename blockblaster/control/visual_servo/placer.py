"""Closed-loop visual-servo placement.

This is the top-level orchestration: open a scrcpy session, press the
finger down on the planned queue slot, coarsely jump toward the target,
then iterate ``detect → check lock → step finger`` until the held piece
is on the suggestion's cells (or we run out of budget).

The interesting algorithmic pieces live in their own modules:

* :mod:`tunables`        — every magic number, each with a comment.
* :mod:`plant_gain`      — online estimator + persistent learned cache.
* :mod:`controller`      — two-mode P controller + Y anti-windup.
* :mod:`detection`       — held-piece detection + lock check.

This file is intentionally just the glue.  If you find yourself adding
new gain math or detection heuristics here, push them down into the
right submodule instead.
"""

from __future__ import annotations

import time
from typing import Optional

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.calibration import CalibrationConfig
from blockblaster.control.coords import (
    grab_to_anchor_offset_px,
    piece_anchor_px,
    slot_center_px,
)
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy
from blockblaster.control.visual_servo import plant_gain
from blockblaster.control.visual_servo.controller import axis_step, clamp_finger_y
from blockblaster.control.visual_servo.detection import (
    detect_piece_cells,
    is_locked,
    piece_anchor as compute_piece_anchor,
    snapshot_initial_placed,
)
from blockblaster.control.visual_servo.tunables import (
    BLIND_COMMIT_TOL_PX,
    FINGER_RENDER_LIFT_PX,
    FRAME_TIMEOUT_S,
    GRAB_Y_NUDGE_PX,
    HOLD_MS,
    INITIAL_LIFT_PX,
    MAX_LOOP_S,
    MAX_NO_PIECE_FRAMES,
    POST_MOVE_SETTLE_MS,
    PRE_LIFT_MS,
    SERVO_DEBUG,
    STABLE_MATCHES,
    ServoResult,
)


def place_with_servo(
    *,
    device:     Device,
    cfg:        CalibrationConfig,
    suggestion: Suggestion,
    frame_w:    int,
    frame_h:    int,
) -> ServoResult:
    """Place ``suggestion.piece`` at ``(row, col)`` via closed-loop servo.

    Holds the finger on the queue slot, drags toward the planned anchor
    while watching the held piece's on-board render, and lifts when the
    cells match.  Returns a :class:`ServoResult` describing the outcome
    so the caller can log / retry.

    Side effect (intentional): on any iteration where we observed valid
    piece motion, the persistent plant-gain cache in
    :mod:`plant_gain` is updated so the *next* placement can size its
    open-loop coarse jump correctly from the first move.
    """
    # ── Pre-flight ──────────────────────────────────────────────────────
    if cfg.grid is None or not cfg.grid.is_valid():
        return ServoResult(False, "grid not calibrated", 0)
    if cfg.queue is None or not cfg.queue.is_valid():
        return ServoResult(False, "queue not calibrated", 0)

    serial = getattr(device, "_serial", None)
    if not serial:
        return ServoResult(False, "device has no ADB serial", 0)

    # Load any previously-learned plant gains for this specific device.
    # First call per process / device hits disk; subsequent calls are no-ops.
    plant_gain.bind_device(serial)

    try:
        dev_w, dev_h = device.screen_size()
    except Exception as exc:
        return ServoResult(False, f"screen_size failed: {exc}", 0)

    dragger = get_scrcpy(serial, dev_w, dev_h)
    if dragger is None:
        return ServoResult(False, "scrcpy control unavailable", 0)

    # Frame → device rescale.  When screenrecord matches `wm size` these
    # are 1.0; otherwise we rescale at the edge so the rest of the file
    # can reason in frame pixels.
    sx = dev_w / max(1, frame_w)
    sy = dev_h / max(1, frame_h)

    def to_dev(p: tuple[int, int]) -> tuple[int, int]:
        return (int(round(p[0] * sx)), int(round(p[1] * sy)))

    # ── Targets ────────────────────────────────────────────────────────
    expected_cells: set[tuple[int, int]] = {
        (suggestion.row + dr, suggestion.col + dc)
        for dr, dc in suggestion.piece.cells
    }
    target_anchor = piece_anchor_px(
        cfg.grid, suggestion.piece, suggestion.row, suggestion.col,
    )
    # Where the *finger* needs to be for the held piece to overlay
    # ``target_anchor``.  Two corrections to ``target_anchor``:
    #
    # 1. **Render lift** (Y only): Block Blast draws the held piece
    #    above the finger by a roughly constant FINGER_RENDER_LIFT_PX.
    #    Without this, top-row placements drive the finger off-screen
    #    above the board and learning never converges (the piece falls
    #    outside the scanner's grid box, see tunables.py).
    # 2. **Grab → anchor offset** (per-piece): the finger grabs the
    #    piece at its geometric centre, but ``target_anchor`` refers to
    #    the bottom-row-centre.  For a vertical 4×1 those points differ
    #    by 1.5 cells in Y; for a horizontal 1×4 they coincide.  Without
    #    this correction the servo over-aims for tall pieces and
    #    under-aims for wide pieces with offset bottoms (L-shapes, etc.).
    grab_dx, grab_dy = grab_to_anchor_offset_px(cfg.grid, suggestion.piece)
    finger_target = (
        target_anchor[0] - grab_dx,
        target_anchor[1] + FINGER_RENDER_LIFT_PX - grab_dy,
    )

    slot_cx, slot_cy = slot_center_px(cfg.queue, suggestion.slot)
    # The piece icon is rendered above the slot's geometric centre; press
    # up there so Block Blast actually picks the piece up on DOWN.
    down_px = (slot_cx, slot_cy - GRAB_Y_NUDGE_PX)
    finger_fpx = down_px

    # ── Snapshot the board so we can isolate the held piece later ──────
    pre_frame, _ = device.get_latest_with_id()
    initial_placed_cells = snapshot_initial_placed(pre_frame, cfg.grid)
    if SERVO_DEBUG:
        print(
            f"[servo init] piece={suggestion.piece.name} "
            f"({suggestion.piece.rows}x{suggestion.piece.cols}) "
            f"initial_placed={len(initial_placed_cells)} cells "
            f"target_cells={sorted(expected_cells)} "
            f"target_anchor={target_anchor} "
            f"grab_offset=({grab_dx},{grab_dy}) "
            f"finger_target={finger_target} grab={down_px}"
        )

    # ── Per-call plant-gain estimate (seeded from cache) ───────────────
    plant_gx, plant_gy = plant_gain.seed_estimates()
    prev_finger:       Optional[tuple[int, int]] = None
    prev_piece_anchor: Optional[tuple[int, int]] = None

    session = None
    iters    = 0
    no_piece = 0
    stable   = 0

    try:
        session = dragger.open_session()

        # ── DOWN on the queue slot, then HOLD still ────────────────────
        # Block Blast registers the grab from a stationary long-press; if
        # we MOVE before the hold elapses, it's read as a swipe and the
        # piece never lifts off the tray.
        session.down(*to_dev(finger_fpx))
        time.sleep(HOLD_MS / 1000.0)

        # ── Tiny initial drift so the piece visibly lifts above the
        # finger (Block Blast renders the held piece offset upward).
        finger_fpx = (finger_fpx[0], finger_fpx[1] - INITIAL_LIFT_PX)
        session.move(*to_dev(finger_fpx))
        time.sleep(POST_MOVE_SETTLE_MS / 1000.0)

        # ── Coarse open-loop jump ──────────────────────────────────────
        # Ideal fraction is 1/plant_gain — de-rated for a small undershoot
        # via plant_gain.coarse_undershoot_for().  On the first placement
        # of a session this falls back to COARSE_FALLBACK; on subsequent
        # placements it uses the per-axis learned plant gain so the jump
        # lands close to the target on the very first move.
        learned_gx, learned_gy = plant_gain.get_learned()
        undershoot_x = plant_gain.coarse_undershoot_for(learned_gx)
        undershoot_y = plant_gain.coarse_undershoot_for(learned_gy)
        coarse_x = finger_fpx[0] + int(round(undershoot_x * (finger_target[0] - finger_fpx[0])))
        coarse_y = finger_fpx[1] + int(round(undershoot_y * (finger_target[1] - finger_fpx[1])))
        finger_fpx = (coarse_x, coarse_y)
        if SERVO_DEBUG:
            print(
                f"[servo coarse] undershoot=({undershoot_x:.2f},{undershoot_y:.2f}) "
                f"learned_plant=("
                f"{learned_gx if learned_gx is not None else 'init'},"
                f"{learned_gy if learned_gy is not None else 'init'})"
            )
        session.move(*to_dev(finger_fpx))
        time.sleep(POST_MOVE_SETTLE_MS / 1000.0)

        # ── Closed-loop servo ──────────────────────────────────────────
        loop_deadline = time.monotonic() + MAX_LOOP_S
        _, last_fid = device.get_latest_with_id()

        while time.monotonic() < loop_deadline:
            iters += 1
            frame, fid = _wait_fresh_frame(device, last_fid, FRAME_TIMEOUT_S)
            if frame is None:
                no_piece += 1
                if no_piece >= MAX_NO_PIECE_FRAMES:
                    return _abort(session, "no frames from device", iters, down_px, to_dev)
                continue
            last_fid = fid

            piece_cells = detect_piece_cells(frame, cfg.grid, initial_placed_cells)
            if not piece_cells:
                if SERVO_DEBUG:
                    # Diagnose detection failures: print BOTH the
                    # placed-band scan and the ghost-band scan so we can
                    # see whether the piece is invisible to both signals
                    # (grab failed / piece off-board) or just one (bad
                    # HSV thresholds for this device).
                    from blockblaster.assist.scanner import scan_board_with_ghost
                    p, g = scan_board_with_ghost(frame, cfg.grid)
                    pc = {(int(r), int(c)) for r, c in zip(*p.nonzero())}
                    gc = {(int(r), int(c)) for r, c in zip(*g.nonzero())}
                    print(
                        f"[servo {iters:02d}] no piece: placed={len(pc)} "
                        f"ghost={len(gc)} initial={len(initial_placed_cells)} "
                        f"new={sorted((pc | gc) - initial_placed_cells)} "
                        f"occluded={sorted(initial_placed_cells - pc)} "
                        f"finger={finger_fpx}"
                    )
                    # On the first failure of a placement, dump the
                    # cropped board to disk so we can eyeball where the
                    # piece actually is vs. where the scanner is looking.
                    # See blockblaster/assist/scanner.py:write_ghost_overlay
                    # for the format — placed=green outline, ghost=yellow.
                    if no_piece == 0:
                        try:
                            from blockblaster.assist.scanner import write_ghost_overlay
                            import os
                            os.makedirs("servo_debug", exist_ok=True)
                            path = f"servo_debug/no_piece_iter{iters:02d}.png"
                            write_ghost_overlay(frame, cfg.grid, path)
                            print(f"[servo {iters:02d}] dumped overlay → {path}")
                        except Exception as exc:
                            print(f"[servo {iters:02d}] overlay dump failed: {exc}")
                no_piece += 1
                if no_piece >= MAX_NO_PIECE_FRAMES:
                    # Blind commit fallback (for devices where the held
                    # piece is rendered outside the calibrated board area
                    # so the scanner never sees it mid-drag).  Only fire
                    # if the open-loop jump actually parked the finger
                    # near the target — otherwise we'd be committing a
                    # guaranteed misplacement and the auto-loop would
                    # think we succeeded.
                    err_x = abs(finger_fpx[0] - finger_target[0])
                    err_y = abs(finger_fpx[1] - finger_target[1])
                    if err_x <= BLIND_COMMIT_TOL_PX and err_y <= BLIND_COMMIT_TOL_PX:
                        if SERVO_DEBUG:
                            print(
                                f"[servo {iters:02d}] blind commit: lifting at "
                                f"finger={finger_fpx} finger_target={finger_target} "
                                f"err=({err_x},{err_y})"
                            )
                        time.sleep(PRE_LIFT_MS / 1000.0)
                        session.up()
                        return ServoResult(True, "blind commit", iters)
                    if SERVO_DEBUG:
                        print(
                            f"[servo {iters:02d}] blind commit DENIED: "
                            f"finger={finger_fpx} finger_target={finger_target} "
                            f"err=({err_x},{err_y}) > tol={BLIND_COMMIT_TOL_PX} "
                            f"— aborting to queue"
                        )
                    return _abort(
                        session,
                        f"piece never appeared (finger {err_x}/{err_y}px from target)",
                        iters, down_px, to_dev,
                    )
                continue
            no_piece = 0

            anchor = compute_piece_anchor(cfg.grid, piece_cells)
            anchor_dx = target_anchor[0] - anchor[0]
            anchor_dy = target_anchor[1] - anchor[1]

            # Update plant-gain estimates from the (Δfinger, Δpiece) we
            # just observed across the previous → current iteration.
            if prev_finger is not None and prev_piece_anchor is not None:
                plant_gx = plant_gain.update_sample(
                    plant_gx,
                    df=finger_fpx[0] - prev_finger[0],
                    dp=anchor[0]     - prev_piece_anchor[0],
                )
                plant_gy = plant_gain.update_sample(
                    plant_gy,
                    df=finger_fpx[1] - prev_finger[1],
                    dp=anchor[1]     - prev_piece_anchor[1],
                )

            locked, lock_reason = is_locked(
                piece_cells, expected_cells, anchor, target_anchor,
            )
            if locked:
                stable += 1
                if stable >= STABLE_MATCHES:
                    # Settle so the game commits the placement on the
                    # same frame the piece was matching.
                    time.sleep(PRE_LIFT_MS / 1000.0)
                    session.up()
                    return ServoResult(True, lock_reason, iters)
                continue
            stable = 0

            dx = axis_step(anchor_dx, plant_gx)
            dy = axis_step(anchor_dy, plant_gy)

            if SERVO_DEBUG:
                print(
                    f"[servo {iters:02d}] piece={anchor} "
                    f"target={target_anchor} raw=({anchor_dx:+d},{anchor_dy:+d}) "
                    f"step=({dx:+d},{dy:+d}) finger={finger_fpx} "
                    f"plant=({plant_gx:.2f},{plant_gy:.2f})"
                )

            prev_finger       = finger_fpx
            prev_piece_anchor = anchor

            if dx == 0 and dy == 0:
                # Piece anchor is centred on target but the cell-set
                # check above didn't fire — either a shape mismatch we
                # can't fix by translation, or the raw error rounded to
                # zero through the gain (very close to target; treat
                # that as a sub-pixel lock).
                if abs(anchor_dx) <= 2 and abs(anchor_dy) <= 2:
                    stable += 1
                    if stable >= STABLE_MATCHES:
                        time.sleep(PRE_LIFT_MS / 1000.0)
                        session.up()
                        return ServoResult(True, "locked (sub-pixel)", iters)
                    continue
                return _abort(session, "piece shape mismatch", iters, down_px, to_dev)

            new_x = finger_fpx[0] + dx
            new_y = clamp_finger_y(finger_fpx[1] + dy, finger_target[1])
            finger_fpx = (new_x, new_y)
            session.move(*to_dev(finger_fpx))
            time.sleep(POST_MOVE_SETTLE_MS / 1000.0)

        return _abort(session, "servo budget exceeded", iters, down_px, to_dev)

    except Exception as exc:
        return ServoResult(False, f"servo error: {exc}", iters)
    finally:
        # Persist whatever we learned this run so the next placement's
        # coarse jump can size itself correctly.  Only write if we saw
        # at least one valid piece detection (prev_piece_anchor is set).
        if prev_piece_anchor is not None:
            plant_gain.set_learned(plant_gx, plant_gy)
            if SERVO_DEBUG:
                from blockblaster.control.visual_servo.plant_gain import (
                    _params_path,
                )
                print(
                    f"[servo learned] plant=({plant_gx:.2f},{plant_gy:.2f}) "
                    f"→ next coarse undershoot ≈ ("
                    f"{plant_gain.coarse_undershoot_for(plant_gx):.2f},"
                    f"{plant_gain.coarse_undershoot_for(plant_gy):.2f}) "
                    f"saved to {_params_path(serial)}"
                )

        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ── Internal helpers ────────────────────────────────────────────────────

def _wait_fresh_frame(device: Device, last_id: int, timeout_s: float):
    """Block until ``frame_id`` advances or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_id:
            return frame, fid
        time.sleep(0.02)
    return None, last_id


def _abort(
    session,
    reason:  str,
    iters:   int,
    down_px: tuple[int, int],
    to_dev,
) -> ServoResult:
    """Drag the finger back over the queue and lift, so the held piece
    slides home instead of placing wrong.  We can't *guarantee* the game
    treats this as a cancel, but a frame outside the board has no piece
    and Block Blast typically returns the piece.
    """
    try:
        session.move(*to_dev(down_px))
        time.sleep(0.05)
        session.up()
    except Exception:
        pass
    return ServoResult(False, reason, iters)
