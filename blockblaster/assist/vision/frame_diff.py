"""Frame-difference motion tracking for the assist GUI.

Holds a single "latest motion" snapshot and fades it over ``ttl`` seconds so a
piece that has stopped moving still glows where it last was. The matching
pygame panel lives in :mod:`blockblaster.assist.render.frame_diff`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from param import BOARD_SIZE

if TYPE_CHECKING:
    from blockblaster.assist.advisor import Suggestion

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels

DIFF_THRESHOLD = 18
DIM_FACTOR = 0.65
HIGHLIGHT_BGR = np.array([60, 230, 255], dtype=np.float32)  # warm amber
DEFAULT_TTL = 1.0
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MOTION_MIN_PIXELS = 200       # below this the screen is treated as static
EVENT_AREA_FRACTION = 0.15    # fraction of screen flagged as a "big" event
EVENT_HOLD = 0.8              # seconds the event flag stays raised


def suggestion_cell_boxes(
    suggestion: Optional[Suggestion],
    board_bbox: Optional[Bbox],
) -> list[tuple[float, float, float, float]]:
    """Frame-px ``(x, y, w, h)`` of each in-bounds suggested-placement cell."""
    if suggestion is None or board_bbox is None:
        return []
    bx, by, bw, bh = board_bbox
    cw = bw / BOARD_SIZE
    ch = bh / BOARD_SIZE
    boxes: list[tuple[float, float, float, float]] = []
    for dr, dc in suggestion.piece.cells:
        r = suggestion.row + dr
        c = suggestion.col + dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            boxes.append((bx + c * cw, by + r * ch, cw, ch))
    return boxes


class FrameDiffTracker:
    """Tracks frame-to-frame motion and caches the latest motion snapshot.

    Call :meth:`observe` on each new frame and :meth:`compose` every render
    tick. Only the most recent snapshot is kept; :meth:`compose` returns it,
    fading out over ``ttl`` seconds.
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
        self._latest_mask: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._motion_fraction: float = 0.0
        self._event_ts: float = 0.0

        self._suggestion: Optional[Suggestion] = None
        self._board_bbox: Optional[Bbox] = None

        self._cached_compose: Optional[np.ndarray] = None
        self._cached_compose_dirty: bool = True

    def set_suggestion(
        self,
        suggestion: Optional[Suggestion],
        board_bbox: Optional[Bbox],
    ) -> None:
        """Update the held advisor placement.

        A momentary ``None`` (e.g. detection dropping during a drag) is
        ignored so the outline doesn't flicker.
        """
        if suggestion is None:
            return
        if suggestion == self._suggestion and board_bbox == self._board_bbox:
            return
        self._suggestion = suggestion
        self._board_bbox = board_bbox

    def clear_suggestion(self) -> None:
        self._suggestion = None
        self._board_bbox = None

    def clear_event(self) -> None:
        """Drop the motion-event hold so analysis can resume after a servo fail."""
        self._event_ts = 0.0

    @property
    def suggestion(self) -> Optional[Suggestion]:
        return self._suggestion

    @property
    def board_bbox(self) -> Optional[Bbox]:
        return self._board_bbox

    def observe(self, frame_bgr: np.ndarray, now: float) -> None:
        """Ingest a new frame; cache its motion mask if it's a motion event."""
        self._last_frame = frame_bgr
        self._cached_compose_dirty = True
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return

        diff = cv2.absdiff(gray, self._prev_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        self._prev_gray = gray

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
        """Return BGR: darkened last frame with the motion snapshot highlighted.

        Rebuilt only when :meth:`observe` ran or the highlight fade is still
        in flight; otherwise the cached composite is returned.
        """
        if self._last_frame is None:
            return None

        fade_active = (
            self._latest_mask is not None
            and (now - self._latest_ts) < self.ttl
        )
        if not self._cached_compose_dirty and self._cached_compose is not None and not fade_active:
            return self._cached_compose

        base = self._last_frame.astype(np.float32) * self.dim_factor
        if fade_active:
            mask3 = self._latest_mask[:, :, None]  # type: ignore[index]
            out = base * (1.0 - mask3) + HIGHLIGHT_BGR[None, None, :] * mask3
        else:
            out = base
        out = np.clip(out, 0, 255).astype(np.uint8)

        self._cached_compose = out
        self._cached_compose_dirty = False
        return out

    @property
    def has_frame(self) -> bool:
        return self._last_frame is not None

    @property
    def motion_fraction(self) -> float:
        """Fraction (0..1) of the screen that changed in the last observed frame."""
        return self._motion_fraction

    def event_active(self, now: float) -> bool:
        """True while a recent big-area change (row clear, etc.) is in effect."""
        return (now - self._event_ts) < EVENT_HOLD
