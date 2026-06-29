"""Closed-loop visual servo for auto-play.

Drags one tray piece onto the advisor's target cells with scrcpy
(DOWN → MOVE… → UP) under continuous visual feedback. A PD controller closes
the loop on a 5-point error so the piece lands on target regardless of grab
offset, render lift, or device scaling.

Detection is palette-invariant: cache the grayscale board just before DOWN as
a baseline, then per frame ``cv2.absdiff`` against it and localise the moving
blob with ``cv2.matchTemplate`` of the piece silhouette. Release fires when
``scan_board`` sees every footprint cell filled (the game's drag preview) for
``RECON_LOCK_FRAMES`` consecutive frames — same HSV occupancy logic as the
reconstruction panel.

Tuning details live in ``docs/visual-servo.md``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.vision.scanner import BOARD_SIZE, scan_board
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy
from blockblaster.game.pieces import Piece

# Grab / gesture timing
HOLD_MS                  = 240
PRE_LIFT_MS              = 260
INITIAL_LIFT_PX          = 150
MIN_INITIAL_LIFT_PX      = 70
INITIAL_LIFT_SETTLE_MS   = 150
INITIAL_LIFT_SUBSTEPS    = 8
INITIAL_LIFT_SUBSTEP_MS  = 12
START_NOISE_X_PX         = 30

# Loop pacing
MAX_LOOP_S           = 5.0
SETTLE_MS            = 50
FRAME_TIMEOUT_S      = 0.03
MAX_NO_PIECE_FRAMES  = 12

# PD controller
GAIN                 = 0.7
DERIV_GAIN           = 1.8
MAX_STEP_PX          = 50
FINE_STEP_PX         = 10
MOVE_SUBSTEPS        = 4
MOVE_SUBSTEP_MS      = 8

# Approach / lock
APPROACH_RADIUS_PX   = 300
ROI_MARGIN_PX        = APPROACH_RADIUS_PX // 2
RECON_LOCK_FRAMES    = 4
BOUNDARY_TOL_PX      = 30
MATCH_SCORE_MIN      = 0.20

# Detection
DIFF_THRESHOLD       = 25
MORPH_KERNEL_PX      = 7
MIN_MOVED_PX         = 60
PIECE_AREA_FRAC      = 0.22

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels


def _five_points(bbox: Bbox) -> list[tuple[int, int]]:
    """Centre + 4 bbox corners — the reference points aligned by the servo."""
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    cx, cy = (x + x1) // 2, (y + y1) // 2
    return [(cx, cy), (x, y), (x1, y), (x, y1), (x1, y1)]


def _footprint_filled(
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


@dataclass
class ServoDebug:
    """Live snapshot of the servo state for the GUI overlay.

    All bboxes are ``(x, y, w, h)`` in frame pixels.
    """
    target_bbox: Optional[Bbox] = None
    measured_bbox: Optional[Bbox] = None
    target_pts: list[tuple[int, int]] = field(default_factory=list)
    measured_pts: list[tuple[int, int]] = field(default_factory=list)
    observe_bbox: Optional[Bbox] = None
    board_aware: bool = False
    unobserved_cells: list[Bbox] = field(default_factory=list)
    finger_px: Optional[tuple[int, int]] = None
    err_px: tuple[int, int] = (0, 0)
    step_px: tuple[int, int] = (0, 0)
    score: float = 0.0
    locked: bool = False
    status: str = ""


DebugSink = Callable[[Optional[ServoDebug]], None]
LogSink = Callable[[str], None]


_MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX),
)
_TEMPLATE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _board_gray(frame_bgr: np.ndarray, region: Bbox) -> np.ndarray:
    x, y, w, h = region
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((max(1, h), max(1, w)), np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _motion_mask(current_gray: np.ndarray, baseline_gray: np.ndarray) -> np.ndarray:
    if current_gray.shape != baseline_gray.shape:
        return np.zeros_like(current_gray)
    diff = cv2.absdiff(current_gray, baseline_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if MORPH_KERNEL_PX > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return mask


def _make_template(piece: Piece, cell_h: int, cell_w: int) -> np.ndarray:
    key = (piece.piece_id, cell_h, cell_w)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    tpl = np.zeros((piece.rows * cell_h, piece.cols * cell_w), np.uint8)
    for dr, dc in piece.cells:
        tpl[dr * cell_h:(dr + 1) * cell_h, dc * cell_w:(dc + 1) * cell_w] = 255
    _TEMPLATE_CACHE[key] = tpl
    return tpl


def _largest_blob_bbox(
    mask: np.ndarray, origin_x: int, origin_y: int, min_area: int,
) -> Optional[Bbox]:
    """Return the largest connected component's bbox (in frame px) or None."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_area:
        return None
    x = origin_x + int(stats[biggest, cv2.CC_STAT_LEFT])
    y = origin_y + int(stats[biggest, cv2.CC_STAT_TOP])
    w = int(stats[biggest, cv2.CC_STAT_WIDTH])
    h = int(stats[biggest, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h)


def _locate_piece(
    frame_bgr: np.ndarray,
    region: Bbox,
    cell_w: int,
    cell_h: int,
    piece: Piece,
    baseline_gray: np.ndarray,
    search_mask: Optional[np.ndarray],
) -> tuple[Optional[Bbox], float]:
    """Localize the held piece by its motion blob inside ``region``.

    Near the target the caller passes a ``search_mask`` (focus window ∩ empty
    cells) so the glow on filled cells can't be picked up; while traveling it
    passes ``None`` and the full motion mask is used.
    """
    region_x, region_y, _, _ = region
    current_gray = _board_gray(frame_bgr, region)
    search = _motion_mask(current_gray, baseline_gray)

    tpl = _make_template(piece, cell_h, cell_w)

    score = 0.0
    if search.shape[0] >= tpl.shape[0] and search.shape[1] >= tpl.shape[1]:
        result = cv2.matchTemplate(search, tpl, cv2.TM_CCORR_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        score = float(score)
    if score < MATCH_SCORE_MIN:
        return None, score

    if search_mask is not None and search_mask.shape == search.shape:
        search = cv2.bitwise_and(search, search_mask)

    expected_area = len(piece.cells) * cell_h * cell_w
    min_area = max(MIN_MOVED_PX, int(PIECE_AREA_FRAC * expected_area))
    return _largest_blob_bbox(search, region_x, region_y, min_area), score


def _move_smooth(
    session, to_dev, start_xy, end_xy, *,
    steps: Optional[int] = None,
    substep_ms: Optional[int] = None,
) -> None:
    n = max(1, steps if steps is not None else MOVE_SUBSTEPS)
    pause_ms = substep_ms if substep_ms is not None else MOVE_SUBSTEP_MS
    for i in range(1, n + 1):
        t = i / n
        x = int(round(start_xy[0] + (end_xy[0] - start_xy[0]) * t))
        y = int(round(start_xy[1] + (end_xy[1] - start_xy[1]) * t))
        session.move(*to_dev((x, y)))
        if i < n and pause_ms > 0:
            time.sleep(pause_ms / 1000)


def _initial_lift_px(target_bbox: Bbox, grid_bbox: Bbox) -> int:
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


def _wait_frame(device: Device, last_fid: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_fid:
            return frame, fid
        time.sleep(0.005)
    return None, last_fid


def _pd_step(
    err_x: int, err_y: int,
    prev_err_x: Optional[int], prev_err_y: Optional[int],
    near_target: bool,
) -> tuple[int, int]:
    derr_x = 0 if prev_err_x is None else err_x - prev_err_x
    derr_y = 0 if prev_err_y is None else err_y - prev_err_y
    ctrl_x = (err_x + DERIV_GAIN * derr_x) / GAIN
    ctrl_y = (err_y + DERIV_GAIN * derr_y) / GAIN
    cap = FINE_STEP_PX if near_target else MAX_STEP_PX
    return (
        max(-cap, min(cap, int(ctrl_x))),
        max(-cap, min(cap, int(ctrl_y))),
    )


def _boundary_override(
    dx: int, dy: int, measured_bbox: Bbox, grid_bbox: Bbox,
) -> tuple[int, int, bool]:
    """If any piece corner drifts off the board, push it firmly back inward."""
    gx, gy, gw, gh = grid_bbox
    gx1, gy1 = gx + gw, gy + gh
    tol = BOUNDARY_TOL_PX
    corners = _five_points(measured_bbox)[1:]
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


def _unobserved_cells(
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
    Returns ``True`` on a confident, on-target release.
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
    target_pts = _five_points(target_bbox)

    # Detection runs over the entire frame so an edge target or piece isn't
    # clipped by the board; board-cell math stays on grid_bbox.
    obs_region: Bbox = (0, 0, frame_w, frame_h)

    pre_frame, _ = device.get_latest_with_id()
    if pre_frame is None:
        emit("[servo] no pre-grab frame available")
        return False
    baseline_gray = _board_gray(pre_frame, obs_region)
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

    unobserved_cells = _unobserved_cells(
        empty_mask, grid_bbox, (0, 0), cell_w, cell_h,
    )

    try:
        with dragger.open_session() as session:
            session.down(*to_dev(grab_px))
            time.sleep(HOLD_MS / 1000)

            jitter = random.randint(-START_NOISE_X_PX, START_NOISE_X_PX)
            lift_px = _initial_lift_px(target_bbox, grid_bbox)
            finger = (grab_px[0] + jitter, grab_px[1] - lift_px)
            _move_smooth(
                session, to_dev, grab_px, finger,
                steps=INITIAL_LIFT_SUBSTEPS,
                substep_ms=INITIAL_LIFT_SUBSTEP_MS,
            )
            time.sleep(INITIAL_LIFT_SETTLE_MS / 1000)

            deadline = time.monotonic() + MAX_LOOP_S
            no_piece = 0
            recon_streak = 0
            iters = 0
            prev_err_x: Optional[int] = None
            prev_err_y: Optional[int] = None
            _, last_fid = device.get_latest_with_id()

            while time.monotonic() < deadline:
                iters += 1
                frame, fid = _wait_frame(device, last_fid, FRAME_TIMEOUT_S)
                if frame is None:
                    continue
                last_fid = fid

                current_board = scan_board(frame, grid_bbox)
                if _footprint_filled(
                    baseline_board, current_board, piece,
                    suggestion.row, suggestion.col,
                ):
                    recon_streak += 1
                else:
                    recon_streak = 0

                if recon_streak >= RECON_LOCK_FRAMES:
                    publish(ServoDebug(
                        target_bbox=target_bbox, finger_px=finger,
                        observe_bbox=grid_bbox, locked=True,
                        status="BOARD LOCK — RELEASING",
                    ))
                    emit(f"[servo {iters}] BOARD LOCK (recon, {recon_streak} frames)")
                    time.sleep(PRE_LIFT_MS / 1000)
                    session.up()
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

                measured_bbox, score = _locate_piece(
                    frame, obs_region, cell_w_i, cell_h_i, piece,
                    baseline_gray, focus_mask if near_target else None,
                )
                if measured_bbox is None:
                    publish(ServoDebug(
                        target_bbox=target_bbox, finger_px=finger, score=score,
                        observe_bbox=observe_bbox, board_aware=near_target,
                        unobserved_cells=unobserved_cells if near_target else [],
                        status="SEARCHING FOR PIECE…",
                    ))
                    if recon_streak == 0:
                        no_piece += 1
                        if no_piece >= MAX_NO_PIECE_FRAMES:
                            emit(f"[servo] lost piece after {iters} iters; aborting")
                            time.sleep(PRE_LIFT_MS / 1000)
                            session.up()
                            return False
                    continue
                no_piece = 0

                # 5-point correspondence: error is the mean of target_i − measured_i.
                measured_pts = _five_points(measured_bbox)
                err_x = int(sum(t[0] - m[0] for t, m in zip(target_pts, measured_pts)) / 5)
                err_y = int(sum(t[1] - m[1] for t, m in zip(target_pts, measured_pts)) / 5)

                dx, dy = _pd_step(err_x, err_y, prev_err_x, prev_err_y, near_target)
                dx, dy, breached = _boundary_override(dx, dy, measured_bbox, grid_bbox)

                if breached:
                    status = "BOUNDARY HIT — PUSHING BACK"
                elif near_target:
                    status = "FOCUSED — FINE APPROACH"
                else:
                    status = "TRAVELING"

                next_finger = (finger[0] + dx, finger[1] + dy)
                publish(ServoDebug(
                    target_bbox=target_bbox, measured_bbox=measured_bbox,
                    target_pts=target_pts, measured_pts=measured_pts,
                    finger_px=finger, err_px=(err_x, err_y),
                    step_px=(dx, dy), score=score, locked=False,
                    observe_bbox=observe_bbox, board_aware=near_target,
                    unobserved_cells=unobserved_cells if near_target else [],
                    status=status,
                ))
                prev_err_x, prev_err_y = err_x, err_y
                _move_smooth(session, to_dev, finger, next_finger)
                finger = next_finger
                time.sleep(SETTLE_MS / 1000)

            emit(f"[servo] budget exceeded after {iters} iters; lifting in place")
            time.sleep(PRE_LIFT_MS / 1000)
            session.up()
            return False
    finally:
        publish(None)
