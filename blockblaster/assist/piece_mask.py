"""Background-subtraction mask & sub-grid sampling helpers for piece recognition.

Separated from :mod:`piece_recognizer` so the core recognizer stays focused on
pipeline orchestration.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ── Thresholds ───────────────────────────────────────────────────────────────
# We separate the piece from the background by sampling a border ring of the
# slot (always background) and masking pixels that differ from that colour.
# This adapts to any game background without hand-tuning HSV ranges.
BG_BORDER_PX        = 6      # width of border ring used to sample background colour
BG_DIFF_THRESHOLD   = 60     # min sum-of-channel-abs-diff to count as foreground
MIN_FILL_PIXELS     = 200    # min mask pixels before considering slot occupied
MIN_FILL_FRACTION   = 0.005  # also require at least 0.5% of slot area
MIN_BBOX_DIM        = 8      # min height/width of tight bbox
MAX_CELL_RATIO      = 1.40   # max cell_h/cell_w (or inverse) before rejecting
CELL_FILL_THRESHOLD = 0.45   # mean mask value (0-1) above which a sub-cell is "filled"
MIN_MATCH_SCORE     = 0.60   # min final score for a result to be returned

MORPH_KERNEL        = 3      # opening kernel size (removes salt noise)


def get_binary_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 binary mask of foreground (piece) pixels.

    Uses background subtraction: samples the slot's border ring (which is
    always background because pieces are centred with padding), then masks
    any pixel that differs sufficiently from that background colour.
    """
    h, w = crop_bgr.shape[:2]
    b = min(BG_BORDER_PX, h // 4, w // 4)
    if b < 1:
        return np.zeros((h, w), dtype=np.uint8)

    border_pixels = np.concatenate([
        crop_bgr[:b, :].reshape(-1, 3),
        crop_bgr[-b:, :].reshape(-1, 3),
        crop_bgr[b:-b, :b].reshape(-1, 3),
        crop_bgr[b:-b, -b:].reshape(-1, 3),
    ])
    bg_color = np.median(border_pixels, axis=0).astype(np.int16)

    diff = np.abs(crop_bgr.astype(np.int16) - bg_color).sum(axis=2)
    mask = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def sample_pattern(
    bbox_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
    cell_h: float,
    cell_w: float,
) -> Optional[np.ndarray]:
    """Slice ``bbox_mask`` into ``n_rows × n_cols`` cells; return a bool pattern.

    Returns ``None`` if any sub-cell is empty (degenerate slicing).
    """
    h, w = bbox_mask.shape
    pattern = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = int(r * cell_h)
            y1 = int((r + 1) * cell_h) if r < n_rows - 1 else h
            x0 = int(c * cell_w)
            x1 = int((c + 1) * cell_w) if c < n_cols - 1 else w
            cell = bbox_mask[y0:y1, x0:x1]
            if cell.size == 0:
                return None
            # mean is in 0–255 (mask values); divide for 0–1
            pattern[r, c] = (cell.mean() / 255.0) > CELL_FILL_THRESHOLD
    return pattern


def count_clusters(profile: np.ndarray, threshold_frac: float = 0.3) -> int:
    """Count contiguous runs in the 1D profile that exceed threshold.

    Each cell of the piece produces one plateau in the projection;
    the gaps between cells produce dips below threshold.
    """
    if profile.size == 0 or profile.max() == 0:
        return 0
    thresh = threshold_frac * profile.max()
    above = profile > thresh
    n = 0
    in_run = False
    for v in above:
        if v and not in_run:
            n += 1
            in_run = True
        elif not v and in_run:
            in_run = False
    return n


def tight(grid: np.ndarray) -> np.ndarray:
    """Remove all-False border rows and columns from a bool grid."""
    rows = np.any(grid, axis=1)
    cols = np.any(grid, axis=0)
    if not rows.any():
        return grid
    r0, r1 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
    c0, c1 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())
    return grid[r0:r1, c0:c1]
