"""Frame-difference motion tracking for the assist GUI.

Computes per-pixel motion between consecutive frames and caches the location
of the most recent movement, so that when an item stops moving its last known
position keeps being reported for up to ``ttl`` seconds.

This module is pure numpy/cv2 (no pygame); the panel that draws the result
lives in :mod:`blockblaster.assist.render.frame_diff`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Pixels whose blurred grayscale difference exceeds this are treated as moving.
DIFF_THRESHOLD = 18
# How dark the original frame is shown underneath the highlight (0..1).
DIM_FACTOR = 0.65
# Highlight tint applied to moving pixels (BGR) — warm amber.
HIGHLIGHT_BGR = np.array([60, 230, 255], dtype=np.float32)
# Seconds the latest motion snapshot keeps being highlighted after it occurs.
DEFAULT_TTL = 1.0
# Morphological-open kernel used to erase isolated video-compression speckle.
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Minimum number of moving pixels (after de-speckling) to count as a real
# motion "event". Below this we assume the screen is effectively static, so the
# previous snapshot is kept frozen instead of being overwritten by noise.
MOTION_MIN_PIXELS = 200
# Fraction of the screen that must change at once to flag a "big" event such as
# a row/section clear (lots of flashing text + effects).
EVENT_AREA_FRACTION = 0.15
# Seconds the "event detected" flag stays raised after the last big-motion frame.
EVENT_HOLD = 0.8

# Board is a fixed 8x8 grid; used to map a suggested (row, col) to pixels.
BOARD_SIZE = 8
# Outline colour baked in for the advisor's suggested placement (BGR) — gold.
SUGGEST_BGR = (40, 200, 255)
# Outline thickness, in pixels, of the baked suggestion footprint.
SUGGEST_OUTLINE_W = 2


def suggestion_cell_boxes(
    suggestion: Optional[object],
    board_bbox: Optional[tuple[int, int, int, int]],
) -> list[tuple[float, float, float, float]]:
    """Pixel-space ``(x, y, w, h)`` of each suggested placement cell on the frame.

    Maps the advisor's ``(row, col)`` placement onto the detected board's
    on-screen bounding box. Shared by the numpy compositor here and the pygame
    overlay in :mod:`blockblaster.assist.render.frame_diff`.
    """
    if suggestion is None or board_bbox is None:
        return []
    bx, by, bw, bh = board_bbox
    cw = bw / BOARD_SIZE
    ch = bh / BOARD_SIZE
    boxes: list[tuple[float, float, float, float]] = []
    for dr, dc in suggestion.piece.cells:
        r = suggestion.row + dr
        c = suggestion.col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            continue
        boxes.append((bx + c * cw, by + r * ch, cw, ch))
    return boxes


class FrameDiffTracker:
    """Tracks frame-to-frame motion and caches only the latest motion snapshot.

    Call :meth:`observe` whenever a *new* frame arrives, and :meth:`compose`
    every render tick. Only the single most recent frame in which motion was
    detected is remembered; :meth:`compose` returns that snapshot, fading it out
    over ``ttl`` seconds. Earlier motion is discarded — there is no trail.
    """

    def __init__(
        self,
        ttl: float = DEFAULT_TTL,
        diff_threshold: int = DIFF_THRESHOLD,
        dim_factor: float = DIM_FACTOR,
    ) -> None:
        self.ttl = ttl
        self.diff_threshold = diff_threshold
        self.dim_factor = dim_factor
        self._prev_gray: Optional[np.ndarray] = None
        self._last_frame: Optional[np.ndarray] = None
        self._latest_mask: Optional[np.ndarray] = None  # latest motion snapshot 0..1
        self._latest_ts: float = 0.0
        self._motion_fraction: float = 0.0  # fraction of screen moving last frame
        self._event_ts: float = 0.0         # time of last big-motion event

        # Advisor suggestion baked into the composite as a gold outline.
        self._suggestion: Optional[object] = None
        self._board_bbox: Optional[tuple[int, int, int, int]] = None

    def set_suggestion(
        self,
        suggestion: Optional[object],
        board_bbox: Optional[tuple[int, int, int, int]],
    ) -> None:
        """Update the advisor placement outlined inside :meth:`compose`.

        The last placement is held until a *different* one arrives: a momentary
        ``None`` (e.g. detection dropping while a piece is mid-drag) is ignored,
        so the outline doesn't flicker. ``Suggestion`` is a frozen dataclass, so
        equality alone tells us whether the placement actually changed.
        """
        if suggestion is None:
            return
        if suggestion == self._suggestion and board_bbox == self._board_bbox:
            return
        self._suggestion = suggestion
        self._board_bbox = board_bbox

    def clear_suggestion(self) -> None:
        """Forget the held placement (e.g. when starting a fresh board)."""
        self._suggestion = None
        self._board_bbox = None

    @property
    def suggestion(self) -> Optional[object]:
        """The currently-held advisor placement (or ``None``)."""
        return self._suggestion

    @property
    def board_bbox(self) -> Optional[tuple[int, int, int, int]]:
        """Board bounding box associated with the held suggestion."""
        return self._board_bbox

    def observe(self, frame_bgr: np.ndarray, now: float) -> None:
        """Ingest a new frame; cache its motion mask iff it is a motion event."""
        self._last_frame = frame_bgr
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return

        diff = cv2.absdiff(gray, self._prev_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        self._prev_gray = gray

        # Binary motion mask, de-speckled so compression noise doesn't register.
        moving = (diff >= self.diff_threshold).astype(np.uint8)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, _OPEN_KERNEL)

        moving_px = int(np.count_nonzero(moving))
        self._motion_fraction = moving_px / float(moving.size)

        if moving_px >= MOTION_MIN_PIXELS:
            self._latest_mask = moving.astype(np.float32)
            self._latest_ts = now

        if self._motion_fraction >= EVENT_AREA_FRACTION:
            self._event_ts = now

    def compose(self, now: float) -> Optional[np.ndarray]:
        """Return a BGR image: darkened last frame with motion + suggestion shown.

        The most recent cached motion snapshot is highlighted in amber until it
        is older than ``ttl``. On top of that, the advisor's suggested placement
        (if any) is outlined in gold.
        """
        if self._last_frame is None:
            return None

        base = self._last_frame.astype(np.float32) * self.dim_factor

        age = now - self._latest_ts
        if self._latest_mask is not None and age < self.ttl:
            mask3 = self._latest_mask[:, :, None]
            out = base * (1.0 - mask3) + HIGHLIGHT_BGR[None, None, :] * mask3
        else:
            out = base

        out = np.clip(out, 0, 255).astype(np.uint8)
        self._bake_suggestion(out)
        return out

    def _bake_suggestion(self, out: np.ndarray) -> None:
        """Draw the gold suggestion-footprint outline into ``out`` (BGR uint8)."""
        if self._suggestion is None or self._board_bbox is None:
            return

        # Map each in-bounds piece cell (by board grid coords) to its pixel rect.
        boxes = suggestion_cell_boxes(self._suggestion, self._board_bbox)
        cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for (dr, dc), (x, y, w, h) in zip(self._suggestion.piece.cells, boxes):
            r, c = self._suggestion.row + dr, self._suggestion.col + dc
            x0, y0 = int(round(x)), int(round(y))
            cells[(r, c)] = (x0, y0, x0 + int(round(w)), y0 + int(round(h)))

        # Draw only the perimeter so internal grid lines stay hidden.
        for (r, c), (x0, y0, x1, y1) in cells.items():
            if (r, c - 1) not in cells:
                cv2.line(out, (x0, y0), (x0, y1), SUGGEST_BGR, SUGGEST_OUTLINE_W)
            if (r, c + 1) not in cells:
                cv2.line(out, (x1, y0), (x1, y1), SUGGEST_BGR, SUGGEST_OUTLINE_W)
            if (r - 1, c) not in cells:
                cv2.line(out, (x0, y0), (x1, y0), SUGGEST_BGR, SUGGEST_OUTLINE_W)
            if (r + 1, c) not in cells:
                cv2.line(out, (x0, y1), (x1, y1), SUGGEST_BGR, SUGGEST_OUTLINE_W)

    @property
    def has_frame(self) -> bool:
        return self._last_frame is not None

    @property
    def motion_fraction(self) -> float:
        """Fraction (0..1) of the screen that changed in the last observed frame."""
        return self._motion_fraction

    def event_active(self, now: float) -> bool:
        """True while a recent big-area change (e.g. a row clear) is in effect."""
        return (now - self._event_ts) < EVENT_HOLD
