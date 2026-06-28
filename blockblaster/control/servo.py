"""Closed-loop visual-servo placement for auto-play.

Drives a single piece from the tray onto the advisor's target cells using a
persistent touch gesture (scrcpy DOWN → MOVE… → UP) under continuous visual
feedback — a PD controller closes the loop so the piece *lands* on target
regardless of grab error, render lift, or device-space scaling.

Detection is colour/palette-invariant: cache the grayscale board crop just
before DOWN as a baseline, then per frame ``cv2.absdiff`` against it, threshold
to a binary "moved" mask, and localise with ``cv2.matchTemplate`` of the known
piece silhouette. The held piece is the only thing moving on the board, so the
diff lights up exactly its footprint regardless of colour or ghost-preview
noise. The controller releases (UP) only once every visible cell sits within
``LOCK_TOL_PX`` of its target — i.e. the piece is confirmed in the right place.

Unlike the old version this takes the board region and grab point directly from
the live detector (analyzer ``board_bbox`` + tray piece bbox) — no manual
calibration step.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.vision.scanner import BOARD_SIZE
from blockblaster.control.device import Device
from blockblaster.control.scrcpy_control import get_scrcpy
from blockblaster.game.pieces import Piece

# ── Grab / gesture timing ─────────────────────────────────────────────────
HOLD_MS              = 240   # dwell after DOWN before any MOVE (long-press grab)
PRE_LIFT_MS          = 260   # settle before UP so the game commits the place
INITIAL_LIFT_PX      = 150    # initial upward nudge so the piece pops above finger
START_NOISE_X_PX     = 45    # random ± x jitter on the initial lift, so a retry
                             # doesn't deterministically repeat the same bad path
# ── Loop pacing ───────────────────────────────────────────────────────────
MAX_LOOP_S           = 7.0   # total servo budget per placement
SETTLE_MS            = 50    # sleep after each move() so the next frame settles
FRAME_TIMEOUT_S      = 0.03  # how long to wait for a fresh frame per iter
MAX_NO_PIECE_FRAMES  = 12    # consecutive no-detect frames before aborting
# ── PD controller ─────────────────────────────────────────────────────────
GAIN                 = 0.7   # P term: piece-px per finger-px (smaller = bigger steps)
DERIV_GAIN           = 1.8   # D term: damps overshoot
MAX_STEP_PX          = 70    # per-iter step clamp while travelling (coarse)
FINE_STEP_PX         = 10     # tighter per-iter clamp once within APPROACH_RADIUS
                             # — small careful nudges near the target, no overshoot
MOVE_SUBSTEPS        = 4     # interpolate each step into N touch-MOVEs
MOVE_SUBSTEP_MS      = 8     # spacing between sub-steps
# ── Approach / lock ───────────────────────────────────────────────────────
APPROACH_RADIUS_PX   = 300    # once the *piece* is within this of the target, focus
                             # on a local window around the target (empty cells
                             # only) for a precise lock; until then track with the
                             # full board mask so the travelling piece isn't lost
ROI_MARGIN_PX        = APPROACH_RADIUS_PX // 2    # half-padding of the local focus window around the
                             # target footprint, in frame pixels
OBSERVE_MARGIN_PX    = 10000 # how far past the board (toward the screen edges) the
                             # observed region extends, so an edge target / piece
                             # isn't clipped to the board (clamped to the frame)
LOCK_TOL_PX          = 30    # |err| px tolerance on both axes to release
LOCK_SCORE_MIN       = 0.70  # required template-match score to release
MATCH_SCORE_MIN      = 0.20  # below this, treat frame as "no piece"
# ── Detection ─────────────────────────────────────────────────────────────
DIFF_THRESHOLD       = 25    # per-pixel grayscale abs-diff threshold
MORPH_KERNEL_PX      = 7     # closing kernel: fills holes for a solid blob
MIN_MOVED_PX         = 60    # absolute floor on the largest blob's area
PIECE_AREA_FRAC      = 0.22  # largest blob must cover at least this fraction of
                             # the piece's expected footprint area — scales the
                             # noise floor to the piece so small speckle (well
                             # below a real piece) is never mistaken for it
EDGE_TRIM_PCT        = 3     # percentile trim on the blob extent (robust to the
                             # blob's own ragged anti-aliased edge)

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels


def _five_points(
    x0: float, y0: float, x1: float, y1: float
) -> list[tuple[int, int]]:
    """Centre + 4 bbox corners — the reference points aligned by the servo."""
    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)
    return [
        (cx, cy),
        (int(x0), int(y0)), (int(x1), int(y0)),
        (int(x0), int(y1)), (int(x1), int(y1)),
    ]


@dataclass
class ServoDebug:
    """Live snapshot of what the servo is tracking, for GUI visualization.

    All points are in *frame pixels*. ``measured_px`` are the per-cell motion
    centroids the controller actually averages; ``target_px`` are the cell
    centres it's driving them toward; ``finger_px`` is the current commanded
    finger position.
    """
    target_bbox: Optional[tuple[int, int, int, int]] = None   # (x0,y0,x1,y1) frame px
    measured_bbox: Optional[tuple[int, int, int, int]] = None  # (x0,y0,x1,y1) frame px
    target_pts: list[tuple[int, int]] = field(default_factory=list)    # 5 pts
    measured_pts: list[tuple[int, int]] = field(default_factory=list)  # 5 pts
    observe_bbox: Optional[tuple[int, int, int, int]] = None    # board region read
    board_aware: bool = False   # True once near target: only empty cells observed
    unobserved_cells: list[tuple[int, int, int, int]] = field(default_factory=list)
    finger_px: Optional[tuple[int, int]] = None
    err_px: tuple[int, int] = (0, 0)
    step_px: tuple[int, int] = (0, 0)   # correction applied to the finger this iter
    score: float = 0.0
    locked: bool = False


DebugSink = Callable[[Optional[ServoDebug]], None]


# ── Detection helpers ──────────────────────────────────────────────────────

def _board_gray(frame_bgr: np.ndarray, grid: Bbox) -> np.ndarray:
    x, y, w, h = grid
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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _make_template(piece: Piece, cell_h: int, cell_w: int) -> np.ndarray:
    tpl = np.zeros((piece.rows * cell_h, piece.cols * cell_w), np.uint8)
    for dr, dc in piece.cells:
        tpl[dr * cell_h:(dr + 1) * cell_h, dc * cell_w:(dc + 1) * cell_w] = 255
    return tpl


def _largest_blob_bbox(
    search: np.ndarray, gx: int, gy: int, min_area: int,
) -> Optional[tuple[float, float, float, float]]:
    """Bbox (frame px) of the largest connected component above ``min_area``."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(search, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_area:
        return None
    ys, xs = np.nonzero(labels == biggest)
    x0 = float(gx + np.percentile(xs, EDGE_TRIM_PCT))
    x1 = float(gx + np.percentile(xs, 100 - EDGE_TRIM_PCT))
    y0 = float(gy + np.percentile(ys, EDGE_TRIM_PCT))
    y1 = float(gy + np.percentile(ys, 100 - EDGE_TRIM_PCT))
    return (x0, y0, x1, y1)


def _locate_piece(
    frame_bgr: np.ndarray,
    region: Bbox,
    cell_w: int,
    cell_h: int,
    piece: Piece,
    baseline_gray: np.ndarray,
    search_mask: Optional[np.ndarray],
) -> tuple[Optional[tuple[float, float, float, float]], float]:
    """Localize the held piece by the *edges* of its largest motion blob.

    Detection runs over ``region`` (the board expanded toward the screen edges,
    so a piece near a board edge isn't clipped), while ``cell_w/cell_h`` are the
    *board* cell pixel sizes used for the shape template. Returns
    ``((x0, y0, x1, y1), match_score)`` in *frame pixels*.

    The caller chooses the mask via the distance gate: while travelling it
    passes ``search_mask=None`` (full motion mask, so the piece isn't ignored);
    near the target it passes the focus mask and we search *strictly* within it
    — no fallback — so a near-complete row's glow can't be picked up instead.
    """
    rx, ry = region[0], region[1]
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

    # Restrict to the focus region/empty cells when asked (near target) —
    # strictly, no fallback, so the glow on filled cells is never chosen.
    if search_mask is not None and search_mask.shape == search.shape:
        search = cv2.bitwise_and(search, search_mask)

    # Noise floor scaled to the piece: a real held piece fills a sizeable
    # fraction of its footprint; tiny flicker blobs fall well under this.
    expected_area = len(piece.cells) * cell_h * cell_w
    min_area = max(MIN_MOVED_PX, int(PIECE_AREA_FRAC * expected_area))
    return _largest_blob_bbox(search, rx, ry, min_area), score


# ── Motion / frame helpers ──────────────────────────────────────────────────

def _move_smooth(session, to_dev, start_xy, end_xy) -> None:
    steps = max(1, MOVE_SUBSTEPS)
    for i in range(1, steps + 1):
        t = i / steps
        x = int(round(start_xy[0] + (end_xy[0] - start_xy[0]) * t))
        y = int(round(start_xy[1] + (end_xy[1] - start_xy[1]) * t))
        session.move(*to_dev((x, y)))
        if i < steps and MOVE_SUBSTEP_MS > 0:
            time.sleep(MOVE_SUBSTEP_MS / 1000)


def _wait_frame(device: Device, last_fid: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_fid:
            return frame, fid
        time.sleep(0.005)
    return None, last_fid


# ── Public entrypoint ───────────────────────────────────────────────────────

def place(
    *,
    device: Device,
    grid_bbox: Bbox,
    grab_px: tuple[int, int],
    suggestion: Suggestion,
    frame_w: int,
    frame_h: int,
    on_debug: Optional[DebugSink] = None,
) -> bool:
    """DOWN on the tray piece, servo it onto the suggested cells, UP.

    ``grid_bbox`` is the board region in frame pixels (analyzer ``board_bbox``)
    and ``grab_px`` is where to press to pick the piece up (centre of the
    detected tray piece). ``on_debug`` (if given) receives a :class:`ServoDebug`
    each iteration for live GUI visualization, and ``None`` on exit. Returns
    ``True`` on a confident, on-target release.
    """
    def _publish(dbg: Optional[ServoDebug]) -> None:
        if on_debug is not None:
            try:
                on_debug(dbg)
            except Exception:  # noqa: BLE001 — debug must never break the servo
                pass
    serial = getattr(device, "_serial", None)
    if not serial:
        print("[servo] device has no serial; cannot inject input")
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

    gx, gy, gw, gh = grid_bbox
    cell_w = gw / BOARD_SIZE
    cell_h = gh / BOARD_SIZE
    cell_w_i = max(1, int(round(cell_w)))
    cell_h_i = max(1, int(round(cell_h)))
    # Target footprint bounding box (edges) of the placement, in frame px.
    piece = suggestion.piece
    tgt_x0 = gx + suggestion.col * cell_w
    tgt_y0 = gy + suggestion.row * cell_h
    tgt_x1 = tgt_x0 + piece.cols * cell_w
    tgt_y1 = tgt_y0 + piece.rows * cell_h
    target_bbox = (int(tgt_x0), int(tgt_y0), int(tgt_x1), int(tgt_y1))
    target_pts = _five_points(tgt_x0, tgt_y0, tgt_x1, tgt_y1)

    # Observed region: the board expanded toward the screen edges, so a piece (or
    # a target) near a board edge isn't clipped. All detection (baseline, motion,
    # masks, blob coords) runs in this region; board cell math stays on grid_bbox.
    ox0 = max(0, gx - OBSERVE_MARGIN_PX)
    oy0 = max(0, gy - OBSERVE_MARGIN_PX)
    ox1 = min(frame_w, gx + gw + OBSERVE_MARGIN_PX)
    oy1 = min(frame_h, gy + gh + OBSERVE_MARGIN_PX)
    obs_region = (ox0, oy0, ox1 - ox0, oy1 - oy0)

    pre_frame, _ = device.get_latest_with_id()
    if pre_frame is None:
        print("[servo] no pre-grab frame available")
        return False
    baseline_gray = _board_gray(pre_frame, obs_region)
    _, empty_mask = cv2.threshold(
        baseline_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Local focus window around the target footprint (frame px), clamped to the
    # observed region (which reaches toward the screen edges) — so the window can
    # extend beyond the board when the target sits on an edge. Near the target we
    # search only here (∩ empty cells).
    roi_x0 = max(ox0, int(tgt_x0) - ROI_MARGIN_PX)
    roi_y0 = max(oy0, int(tgt_y0) - ROI_MARGIN_PX)
    roi_x1 = min(ox1, int(tgt_x1) + ROI_MARGIN_PX)
    roi_y1 = min(oy1, int(tgt_y1) + ROI_MARGIN_PX)
    roi_frame = (roi_x0, roi_y0, roi_x1, roi_y1)
    roi_mask = np.zeros_like(empty_mask)
    roi_mask[roi_y0 - oy0:roi_y1 - oy0, roi_x0 - ox0:roi_x1 - ox0] = 255
    focus_mask = cv2.bitwise_and(empty_mask, roi_mask)

    # Per-cell split (for the debug overlay): cells that are mostly filled are
    # the ones board-aware mode stops observing near the target. Sampled from
    # empty_mask at obs-region-relative coordinates.
    unobserved_cells: list[tuple[int, int, int, int]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            fx0, fx1 = gx + int(c * cell_w), gx + int((c + 1) * cell_w)
            fy0, fy1 = gy + int(r * cell_h), gy + int((r + 1) * cell_h)
            cell = empty_mask[fy0 - oy0:fy1 - oy0, fx0 - ox0:fx1 - ox0]
            if cell.size and float(np.count_nonzero(cell)) / cell.size < 0.5:
                unobserved_cells.append(
                    (fx0, fy0, fx1, fy1)
                )

    try:
        with dragger.open_session() as session:
            session.down(*to_dev(grab_px))
            time.sleep(HOLD_MS / 1000)

            # Dither the initial x so a retry doesn't repeat the same path.
            jitter = random.randint(-START_NOISE_X_PX, START_NOISE_X_PX)
            finger = (grab_px[0] + jitter, grab_px[1] - INITIAL_LIFT_PX)
            _move_smooth(session, to_dev, grab_px, finger)
            time.sleep(SETTLE_MS / 1000)

            deadline = time.monotonic() + MAX_LOOP_S
            no_piece = 0
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

                # Board-aware glow suppression only kicks in near the target;
                # while travelling we track with the full motion mask so the
                # piece (passing over filled cells) is never ignored.
                # Focus on the local target window (empty cells only) once the
                # *piece* (last measured error) is close; while it's still far /
                # unmeasured, scan the whole board so the travelling piece isn't
                # lost. Gating on the piece error — not the finger — avoids the
                # render-lift offset between finger and piece.
                near_target = (
                    prev_err_x is not None
                    and (prev_err_x ** 2 + prev_err_y ** 2) <= APPROACH_RADIUS_PX ** 2
                )
                observe_kw = dict(
                    observe_bbox=roi_frame if near_target else grid_bbox,
                    board_aware=near_target,
                    unobserved_cells=unobserved_cells if near_target else [],
                )
                measured_bbox, score = _locate_piece(
                    frame, obs_region, cell_w_i, cell_h_i, suggestion.piece,
                    baseline_gray, focus_mask if near_target else None,
                )
                if measured_bbox is None:
                    _publish(ServoDebug(
                        target_bbox=target_bbox, finger_px=finger, score=score,
                        **observe_kw,
                    ))
                    no_piece += 1
                    if no_piece >= MAX_NO_PIECE_FRAMES:
                        print(f"[servo] lost piece after {iters} iters; aborting")
                        time.sleep(PRE_LIFT_MS / 1000)
                        session.up()
                        return False
                    continue
                no_piece = 0

                mx0, my0, mx1, my1 = measured_bbox
                measured_bbox_int = (int(mx0), int(my0), int(mx1), int(my1))
                # 5-point correspondence (centre + 4 corners): the error is the
                # mean of target_i − measured_i over all five reference points.
                measured_pts = _five_points(mx0, my0, mx1, my1)
                err_x = int(sum(t[0] - m[0] for t, m in zip(target_pts, measured_pts)) / 5)
                err_y = int(sum(t[1] - m[1] for t, m in zip(target_pts, measured_pts)) / 5)

                locked = (abs(err_x) <= LOCK_TOL_PX and abs(err_y) <= LOCK_TOL_PX
                          and score >= LOCK_SCORE_MIN)

                if locked:
                    _publish(ServoDebug(
                        target_bbox=target_bbox, measured_bbox=measured_bbox_int,
                        target_pts=target_pts, measured_pts=measured_pts,
                        finger_px=finger, err_px=(err_x, err_y),
                        score=score, locked=True, **observe_kw,
                    ))
                    print(f"[servo {iters}] LOCK err=({err_x:+d},{err_y:+d}) "
                          f"score={score:.2f}")
                    time.sleep(PRE_LIFT_MS / 1000)
                    session.up()
                    return True

                derr_x = 0 if prev_err_x is None else err_x - prev_err_x
                derr_y = 0 if prev_err_y is None else err_y - prev_err_y
                ctrl_x = (err_x + DERIV_GAIN * derr_x) / GAIN
                ctrl_y = (err_y + DERIV_GAIN * derr_y) / GAIN
                # Careful near the target: cap each axis tighter once focused.
                cap = FINE_STEP_PX if near_target else MAX_STEP_PX
                dx = max(-cap, min(cap, int(ctrl_x)))
                dy = max(-cap, min(cap, int(ctrl_y)))
                next_finger = (finger[0] + dx, finger[1] + dy)
                _publish(ServoDebug(
                    target_bbox=target_bbox, measured_bbox=measured_bbox_int,
                    target_pts=target_pts, measured_pts=measured_pts,
                    finger_px=finger, err_px=(err_x, err_y),
                    step_px=(dx, dy), score=score, locked=False, **observe_kw,
                ))
                print(f"[servo {iters}] err=({err_x:+d},{err_y:+d}) "
                      f"score={score:.2f} step=({dx:+d},{dy:+d})")
                prev_err_x, prev_err_y = err_x, err_y
                _move_smooth(session, to_dev, finger, next_finger)
                finger = next_finger
                time.sleep(SETTLE_MS / 1000)

            print(f"[servo] budget exceeded after {iters} iters; lifting in place")
            time.sleep(PRE_LIFT_MS / 1000)
            session.up()
            return False
    finally:
        _publish(None)
