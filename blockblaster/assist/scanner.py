"""Sample the Block Blast board area and return an 8x8 occupancy grid.

A cell is "filled" when its centre patch has high HSV value AND either high
saturation (coloured block) or extremely high value (white block). Empty cells
are the dark navy background.
"""

from __future__ import annotations

import cv2
import numpy as np

BOARD_SIZE          = 8
SAT_THRESHOLD       = 60
VAL_THRESHOLD       = 130
WHITE_VAL_THRESHOLD = 190

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels


def scan_board(frame_bgr: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Return an (8, 8) bool grid of filled cells inside ``bbox``."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    crop   = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

    target = BOARD_SIZE * 32
    crop_r = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)
    hsv    = cv2.cvtColor(crop_r, cv2.COLOR_BGR2HSV)
    cell   = target // BOARD_SIZE
    half   = cell // 2

    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cy = row * cell + half
            cx = col * cell + half
            patch = hsv[cy - 4 : cy + 5, cx - 4 : cx + 5]
            sat = float(patch[:, :, 1].mean())
            val = float(patch[:, :, 2].mean())
            grid[row, col] = val > VAL_THRESHOLD and (
                sat > SAT_THRESHOLD or val > WHITE_VAL_THRESHOLD
            )
    return grid
