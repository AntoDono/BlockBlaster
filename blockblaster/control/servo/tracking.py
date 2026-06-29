"""Frame-diff piece localization for the visual servo loop."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from blockblaster.control.servo.config import (
    DIFF_THRESHOLD,
    MATCH_SCORE_MIN,
    MIN_MOVED_PX,
    MORPH_KERNEL_PX,
    PIECE_AREA_FRAC,
)
from blockblaster.control.servo.types import Bbox
from blockblaster.game.pieces import Piece

_MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_PX, MORPH_KERNEL_PX),
)
_TEMPLATE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def board_gray(frame_bgr: np.ndarray, region: Bbox) -> np.ndarray:
    x, y, w, h = region
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((max(1, h), max(1, w)), np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def motion_mask(current_gray: np.ndarray, baseline_gray: np.ndarray) -> np.ndarray:
    if current_gray.shape != baseline_gray.shape:
        return np.zeros_like(current_gray)
    diff = cv2.absdiff(current_gray, baseline_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if MORPH_KERNEL_PX > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return mask


def make_template(piece: Piece, cell_h: int, cell_w: int) -> np.ndarray:
    key = (piece.piece_id, cell_h, cell_w)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    tpl = np.zeros((piece.rows * cell_h, piece.cols * cell_w), np.uint8)
    for dr, dc in piece.cells:
        tpl[dr * cell_h:(dr + 1) * cell_h, dc * cell_w:(dc + 1) * cell_w] = 255
    _TEMPLATE_CACHE[key] = tpl
    return tpl


def largest_blob(
    mask: np.ndarray, origin_x: int, origin_y: int, min_area: int,
) -> tuple[Optional[Bbox], int]:
    """Return the largest connected component's bbox and pixel area, or (None, 0)."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    area_px = int(stats[biggest, cv2.CC_STAT_AREA])
    if area_px < min_area:
        return None, 0
    x = origin_x + int(stats[biggest, cv2.CC_STAT_LEFT])
    y = origin_y + int(stats[biggest, cv2.CC_STAT_TOP])
    w = int(stats[biggest, cv2.CC_STAT_WIDTH])
    h = int(stats[biggest, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h), area_px


def largest_blob_bbox(
    mask: np.ndarray, origin_x: int, origin_y: int, min_area: int,
) -> Optional[Bbox]:
    bbox, _ = largest_blob(mask, origin_x, origin_y, min_area)
    return bbox


def locate_piece(
    frame_bgr: np.ndarray,
    region: Bbox,
    cell_w: int,
    cell_h: int,
    piece: Piece,
    baseline_gray: np.ndarray,
    search_mask: Optional[np.ndarray],
) -> tuple[Optional[Bbox], float, int]:
    """Localize the held piece by its motion blob inside ``region``.

    Near the target the caller passes a ``search_mask`` (focus window ∩ empty
    cells) so the glow on filled cells can't be picked up; while traveling it
    passes ``None`` and the full motion mask is used.
    """
    region_x, region_y, _, _ = region
    current_gray = board_gray(frame_bgr, region)
    search = motion_mask(current_gray, baseline_gray)

    tpl = make_template(piece, cell_h, cell_w)

    score = 0.0
    if search.shape[0] >= tpl.shape[0] and search.shape[1] >= tpl.shape[1]:
        result = cv2.matchTemplate(search, tpl, cv2.TM_CCORR_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        score = float(score)
    if score < MATCH_SCORE_MIN:
        return None, score, 0

    if search_mask is not None and search_mask.shape == search.shape:
        search = cv2.bitwise_and(search, search_mask)

    expected_area = len(piece.cells) * cell_h * cell_w
    min_area = max(MIN_MOVED_PX, int(PIECE_AREA_FRAC * expected_area))
    bbox, area_px = largest_blob(search, region_x, region_y, min_area)
    return bbox, score, area_px
