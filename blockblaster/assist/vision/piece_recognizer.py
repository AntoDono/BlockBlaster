"""Piece recognition for Block Blast tray pieces.

Two backends:

* :class:`blockblaster.piece_cnn.PieceClassifier` — preferred, used whenever
  ``piece_cnn.pt`` is available.
* Projection / template-matching fallback (see :mod:`piece_mask`).

The recognizer takes a list of BGR crops (one per detected tray piece) and
returns one ``(piece, confidence)`` per crop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from blockblaster.assist.vision.piece_mask import (
    MAX_CELL_RATIO,
    MIN_BBOX_DIM,
    MIN_FILL_FRACTION,
    MIN_FILL_PIXELS,
    MIN_MATCH_SCORE,
    count_clusters,
    get_binary_mask,
    sample_pattern,
    tight,
)
from blockblaster.game.pieces import PIECES, Piece


class PieceRecognizer:
    """Recognises which Block Blast piece occupies each tray crop."""

    def __init__(self) -> None:
        self._templates: list[tuple[Piece, np.ndarray]] = [
            (p, tight(p.to_grid())) for p in PIECES
        ]
        try:
            from blockblaster.piece_cnn import PieceClassifier
            self._cnn: Optional["PieceClassifier"] = PieceClassifier()
            if not self._cnn.is_ready:
                print(f"[recognizer] CNN unavailable: {self._cnn.last_error}; "
                      "falling back to projection heuristic")
        except Exception as exc:  # noqa: BLE001
            print(f"[recognizer] CNN init error: {exc!r}; using projection heuristic")
            self._cnn = None

    def recognize_crops(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[Optional[Piece], float]]:
        """Classify each BGR crop; returns ``(piece_or_None, confidence)`` per crop."""
        if not crops:
            return []
        if self._cnn is not None and self._cnn.is_ready:
            return self._cnn.classify_slots(crops)
        return [self._recognize_one(c) for c in crops]

    def _recognize_one(self, crop: np.ndarray) -> tuple[Optional[Piece], float]:
        if crop.size == 0:
            return None, 0.0

        mask = get_binary_mask(crop)
        fill = int(np.count_nonzero(mask))
        if fill < MIN_FILL_PIXELS or fill < MIN_FILL_FRACTION * crop.shape[0] * crop.shape[1]:
            return None, 0.0

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None, 0.0
        r0 = int(rows.argmax())
        r1 = int(len(rows) - rows[::-1].argmax())
        c0 = int(cols.argmax())
        c1 = int(len(cols) - cols[::-1].argmax())

        bbox = mask[r0:r1, c0:c1]
        h, w = bbox.shape
        if h < MIN_BBOX_DIM or w < MIN_BBOX_DIM:
            return None, 0.0

        n_rows_proj = count_clusters(bbox.sum(axis=1))
        n_cols_proj = count_clusters(bbox.sum(axis=0))

        candidates: list[tuple[float, Piece]] = []
        if 1 <= n_rows_proj <= 5 and 1 <= n_cols_proj <= 5:
            self._score(bbox, n_rows_proj, n_cols_proj, candidates, weight=1.0)
        if not candidates:
            for n_r in range(1, 6):
                for n_c in range(1, 6):
                    self._score(bbox, n_r, n_c, candidates, weight=0.95)
        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda t: t[0], reverse=True)
        score, piece = candidates[0]
        if score < MIN_MATCH_SCORE:
            return None, score
        return piece, score

    def _score(
        self,
        bbox: np.ndarray,
        n_rows: int,
        n_cols: int,
        out: list[tuple[float, Piece]],
        weight: float,
    ) -> None:
        h, w = bbox.shape
        cell_h = h / n_rows
        cell_w = w / n_cols
        ratio = max(cell_h, cell_w) / max(min(cell_h, cell_w), 1e-6)
        if ratio > MAX_CELL_RATIO:
            return
        squareness = min(cell_h, cell_w) / max(cell_h, cell_w)

        pattern = sample_pattern(bbox, n_rows, n_cols, cell_h, cell_w)
        if pattern is None or not pattern.any():
            return
        if not pattern[0, :].any() or not pattern[-1, :].any():
            return
        if not pattern[:, 0].any() or not pattern[:, -1].any():
            return

        for piece, template in self._templates:
            if template.shape != pattern.shape:
                continue
            match_ratio = float((template == pattern).sum()) / template.size
            out.append((match_ratio * squareness * weight, piece))


def crop_bbox(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Safely crop ``frame_bgr`` to ``(x, y, w, h)`` clamped to the frame."""
    fh, fw = frame_bgr.shape[:2]
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    return frame_bgr[y1:y2, x1:x2]


def pad_to_slot(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg_bgr: np.ndarray,
    fill_frac: float = 0.60,
) -> np.ndarray:
    """Return a square BGR canvas with the bbox crop centred on a bg-coloured pad.

    The CNN was trained on slots where the piece occupies ~40-88% of the slot
    area. Detector bboxes are *tight*, so we paste the crop onto a square
    canvas sized so the piece fills ``fill_frac`` of the canvas edge — bringing
    inputs back in-distribution.
    """
    crop = crop_bbox(frame_bgr, bbox)
    if crop.size == 0:
        return crop

    ch, cw = crop.shape[:2]
    side   = max(int(round(max(ch, cw) / max(fill_frac, 1e-3))), max(ch, cw))

    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[:] = np.clip(bg_bgr, 0, 255).astype(np.uint8)

    y0 = (side - ch) // 2
    x0 = (side - cw) // 2
    canvas[y0:y0 + ch, x0:x0 + cw] = crop
    return canvas
