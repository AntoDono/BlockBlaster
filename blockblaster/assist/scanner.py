"""Scan a BGR frame and return an 8×8 bool grid representing the game board state.

Strategy:
  1. Crop the frame to the calibrated bounding box.
  2. Divide the crop into an 8×8 grid.
  3. Sample the center pixel of each cell in HSV space.
  4. A cell is "filled" if its HSV Value (brightness) > VAL_THRESHOLD AND
     Saturation > SAT_THRESHOLD.

  Block Blast's empty cells are dark navy — low brightness regardless of hue.
  Filled blocks are vivid AND bright. Saturation alone is insufficient because
  the navy background also has noticeable saturation.
"""

from __future__ import annotations

import cv2
import numpy as np

from blockblaster.assist.calibration import CalibrationBox

BOARD_SIZE    = 8
SAT_THRESHOLD = 60    # HSV S 0–255; blocks are vivid
VAL_THRESHOLD = 130   # HSV V 0–255; filled block centres ~200-235, empty cells ~80-92


def scan_board(frame_bgr: np.ndarray, box: CalibrationBox) -> np.ndarray:
    """Return an (8, 8) bool ndarray: True = cell filled, False = empty.

    Returns an all-False grid if the box is invalid or the crop is degenerate.
    """
    if not box.is_valid():
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, box.fx)
    y1 = max(0, box.fy)
    x2 = min(fw, box.fx + box.fw)
    y2 = min(fh, box.fy + box.fh)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

    # Resize to a fixed resolution so cell boundaries are exact integer pixels
    target_px = BOARD_SIZE * 32   # 256×256, divisible by 8
    crop_resized = cv2.resize(crop, (target_px, target_px), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    cell_px = target_px // BOARD_SIZE   # 32 px per cell
    half    = cell_px // 2

    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cy = row * cell_px + half
            cx = col * cell_px + half
            # Sample a 9×9 patch well inside the cell (cell is 32px wide)
            patch = hsv[cy - 4 : cy + 5, cx - 4 : cx + 5]
            sat = float(patch[:, :, 1].mean())   # S channel
            val = float(patch[:, :, 2].mean())   # V channel
            grid[row, col] = sat > SAT_THRESHOLD and val > VAL_THRESHOLD

    return grid
