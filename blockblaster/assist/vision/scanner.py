"""Sample the Block Blast board area and return an 8x8 occupancy grid.

A cell is "filled" when its centre patch has high HSV value AND either high
saturation (coloured block) or extremely high value (white block). Empty cells
are the dark navy background.
"""

from __future__ import annotations

import cv2
import numpy as np

from param import BOARD_SIZE  # re-exported for compat with control/servo etc.


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

    # Vectorised 9×9 centre-patch sample for all 64 cells: reshape the HSV crop
    # into an (8, 8, cell, cell, 3) block grid and slice each block's centre.
    blocks = hsv.reshape(BOARD_SIZE, cell, BOARD_SIZE, cell, 3).swapaxes(1, 2)
    centres = blocks[:, :, half - 4:half + 5, half - 4:half + 5, :]
    sat = centres[..., 1].mean(axis=(2, 3))
    val = centres[..., 2].mean(axis=(2, 3))
    return (val > VAL_THRESHOLD) & ((sat > SAT_THRESHOLD) | (val > WHITE_VAL_THRESHOLD))
