"""Recognise Block Blast pieces from cropped slot images.

Strategy
--------
1. For each of the 3 queue slots, crop from the calibrated queue box.
2. Apply an HSV brightness+saturation threshold to get a binary filled mask
   (same constants as scanner.py so both modules stay in sync).
3. Find the tight bounding box of the filled region.
4. Resize the bounding box to a fixed 32×32 binary template.
5. Compare against 32 pre-built piece templates using Intersection-over-Union.
6. Return the Piece with the highest IoU (or None if the slot is empty).

All comparison is on 32×32 binary arrays — ~100k float ops total per frame, well
under 1ms with vectorised numpy.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.game.pieces import PIECES, Piece

# ── Thresholds (kept in sync with scanner.py) ────────────────────────────────
SAT_THRESHOLD = 60    # HSV S 0-255
VAL_THRESHOLD = 130   # HSV V 0-255; filled blocks are bright, empty cells dark

# ── Template matching ─────────────────────────────────────────────────────────
TEMPLATE_PX    = 32   # normalised resolution for both templates and observations
MIN_FILL_RATIO = 0.02 # fraction of slot pixels that must be filled to count

# Maximum piece dimensions across all 32 pieces (used for size normalisation)
_MAX_ROWS = max(p.rows for p in PIECES)
_MAX_COLS = max(p.cols for p in PIECES)

# Weights for combined scoring: shape (IoU) + size (area fraction)
_IOU_WEIGHT  = 0.65
_SIZE_WEIGHT = 0.35


class PieceRecognizer:
    """Pre-builds 32×32 shape templates for all canonical pieces at init time."""

    def __init__(self) -> None:
        # Each entry: (Piece, 32×32 binary template, normalised size ratio)
        self._templates: list[tuple[Piece, np.ndarray, float]] = self._build_templates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_queue(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
    ) -> list[Optional[Piece]]:
        """Return a length-3 list of recognised Pieces (None = empty slot).

        The queue box is split into 3 equal horizontal columns.
        """
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, queue_box.fx)
        y1 = max(0, queue_box.fy)
        x2 = min(fw, queue_box.fx + queue_box.fw)
        y2 = min(fh, queue_box.fy + queue_box.fh)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return [None, None, None]

        slot_w  = crop.shape[1] // 3
        slot_h  = crop.shape[0]
        results: list[Optional[Piece]] = []
        for i in range(3):
            sx1 = i * slot_w
            sx2 = sx1 + slot_w if i < 2 else crop.shape[1]
            slot = crop[:, sx1:sx2]
            results.append(self._recognize_slot(slot, slot_h, slot_w))

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recognize_slot(
        self,
        slot_bgr: np.ndarray,
        slot_h: int,
        slot_w: int,
    ) -> Optional[Piece]:
        """Match one slot crop to the closest canonical piece, or return None."""
        if slot_bgr.size == 0:
            return None

        # Step 1: HSV binary mask
        hsv  = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 1].astype(np.int32) > SAT_THRESHOLD) &
            (hsv[:, :, 2].astype(np.int32) > VAL_THRESHOLD)
        ).astype(np.uint8)

        # Step 2: Reject empty slots
        fill_ratio = mask.sum() / mask.size
        if fill_ratio < MIN_FILL_RATIO:
            return None

        # Step 3: Tight bounding box via OpenCV
        nz = cv2.findNonZero(mask)
        if nz is None:
            return None
        bx, by, bw, bh = cv2.boundingRect(nz)
        if bw == 0 or bh == 0:
            return None

        # Step 4: Shape feature — aspect-ratio-preserving resize to TEMPLATE_PX×TEMPLATE_PX
        roi     = mask[by : by + bh, bx : bx + bw].astype(np.float32)
        obs_bin = PieceRecognizer._fit_to_canvas(roi)

        # Step 5: Size feature — fraction of slot area covered by the bounding box
        obs_size = (bh * bw) / max(slot_h * slot_w, 1)

        # Step 6: Combined score against all 32 templates
        best_piece: Optional[Piece] = None
        best_score = -1.0

        for piece, tmpl, tmpl_size in self._templates:
            # Shape similarity (IoU)
            intersection = float((obs_bin * tmpl).sum())
            union        = float(obs_bin.sum() + tmpl.sum() - intersection)
            iou          = intersection / union if union > 0 else 0.0

            # Size similarity (1 = perfect match, 0 = maximally different)
            size_sim = max(0.0, 1.0 - abs(obs_size - tmpl_size))

            score = _IOU_WEIGHT * iou + _SIZE_WEIGHT * size_sim
            if score > best_score:
                best_score = score
                best_piece = piece

        return best_piece

    @staticmethod
    def _build_templates() -> list[tuple[Piece, np.ndarray, float]]:
        """Pre-compute (Piece, 32×32 template, expected size ratio) for every piece."""
        templates = []
        for piece in PIECES:
            grid     = piece.to_grid().astype(np.float32)
            tmpl_bin = PieceRecognizer._fit_to_canvas(grid)
            # Normalised expected size: fraction of a (_MAX_ROWS × _MAX_COLS) slot
            # that the piece's bounding box would occupy.
            size_ratio = (piece.rows * piece.cols) / (_MAX_ROWS * _MAX_COLS)
            templates.append((piece, tmpl_bin, size_ratio))
        return templates

    @staticmethod
    def _fit_to_canvas(grid: np.ndarray) -> np.ndarray:
        """Resize a 2-D float array into a TEMPLATE_PX×TEMPLATE_PX canvas while
        preserving aspect ratio (letter-box / pillar-box with zero padding)."""
        h, w = grid.shape
        scale = min(TEMPLATE_PX / h, TEMPLATE_PX / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(
            grid.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((TEMPLATE_PX, TEMPLATE_PX), dtype=np.float32)
        pad_y = (TEMPLATE_PX - new_h) // 2
        pad_x = (TEMPLATE_PX - new_w) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return (canvas > 0.5).astype(np.float32)
