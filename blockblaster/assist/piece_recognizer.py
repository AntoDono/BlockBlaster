"""Piece recognition for the Block Blast queue.

Two backends are wired up here:

1. **CNN classifier** (preferred) — :class:`blockblaster.piece_cnn.PieceClassifier`
   trained on synthetic data.  Used whenever ``piece_cnn.pt`` is available.
2. **Heuristic fallback** — sub-grid sampling against piece templates.  Used
   automatically when the CNN weights are missing.

Heuristic pipeline (per slot):
  1. Crop one of the 3 equal horizontal thirds from the queue bounding box.
  2. Background-subtract → binary mask (see :mod:`piece_mask`).
  3. If the mask is too sparse → empty slot.
  4. Tight-bbox the mask to trim padding around a centred piece.
  5. Try every plausible ``(n_rows, n_cols) ∈ {1..5}²`` sub-division.  Each
     candidate must have a near-square cell aspect and a tight pattern
     (filled cells touching every edge).
  6. Match each candidate pattern against piece templates that share its
     shape; score = match-ratio × squareness × weight.
  7. Return the highest-scoring piece (or ``None`` below the min confidence).

Why this beats blob detection: it never tries to separate adjacent cells.
Touching cells, fragmented cells from highlights/shadows, and weird mask
gradients are all OK — every candidate sub-division re-tiles the bbox cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.assist.piece_debug import (
    SlotDebug,
    format_slot_report,
    render_overlay,
)
from blockblaster.assist.piece_mask import (
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
    """Recognises which Block Blast piece occupies each queue slot.

    By default this uses a CNN classifier trained on synthetic data
    (``piece_cnn.pt``).  If the CNN weights are missing or fail to load it
    transparently falls back to the legacy projection / template-matching
    heuristic so the assist GUI still works.
    """

    def __init__(self) -> None:
        # Pre-compute tight boolean grid for every piece (template lookup table)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_queue(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
    ) -> list[Optional[Piece]]:
        """Return a 3-element list of recognised pieces (None if unrecognised)."""
        return [p for p, _ in self.recognize_queue_with_confidence(frame_bgr, queue_box)]

    def recognize_queue_with_confidence(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
    ) -> list[tuple[Optional[Piece], float]]:
        """Like :meth:`recognize_queue` but also returns a confidence in [0, 1].

        For the CNN path this is the softmax probability of the chosen class.
        For the heuristic fallback it's the template-match score.
        """
        slot_crops = list(self._iter_slot_crops(frame_bgr, queue_box))
        if self._cnn is not None and self._cnn.is_ready:
            return self._cnn.classify_slots(slot_crops)
        return [
            (p, dbg.best_score)
            for p, dbg in self._recognize_queue_with_debug(frame_bgr, queue_box)
        ]

    def save_debug(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
        out_dir: Path | str = "assist_debug",
    ) -> Path:
        """Run recognition and dump per-slot diagnostic images + summary.

        Returns the output directory path.  Files written per slot:
            slot_{i}_crop.png        — raw BGR slot crop
            slot_{i}_mask.png        — binary mask
            slot_{i}_overlay.png     — slot + bbox + sub-grid lines
            summary.txt              — text report for all 3 slots
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        slot_results = self._recognize_queue_with_debug(frame_bgr, queue_box)

        report: list[str] = []
        slots = self._iter_slot_crops(frame_bgr, queue_box)
        for i, (slot_crop, (piece, dbg)) in enumerate(zip(slots, slot_results)):
            slot_name = f"slot_{i + 1}"
            cv2.imwrite(str(out_path / f"{slot_name}_crop.png"), slot_crop)

            mask = get_binary_mask(slot_crop)
            cv2.imwrite(str(out_path / f"{slot_name}_mask.png"), mask)

            overlay = render_overlay(slot_crop, dbg)
            cv2.imwrite(str(out_path / f"{slot_name}_overlay.png"), overlay)

            report.append(format_slot_report(i, piece, dbg))

        (out_path / "summary.txt").write_text("\n\n".join(report))
        return out_path

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _recognize_queue_with_debug(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
    ) -> list[tuple[Optional[Piece], SlotDebug]]:
        return [
            self._recognize_slot(crop)
            for crop in self._iter_slot_crops(frame_bgr, queue_box)
        ]

    @staticmethod
    def _iter_slot_crops(
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
    ):
        fh, fw = frame_bgr.shape[:2]
        x0 = max(0, queue_box.fx)
        y0 = max(0, queue_box.fy)
        x1 = min(fw, queue_box.fx + queue_box.fw)
        y1 = min(fh, queue_box.fy + queue_box.fh)

        slot_w = (x1 - x0) // 3
        for i in range(3):
            sx0 = x0 + i * slot_w
            sx1 = x0 + (i + 1) * slot_w if i < 2 else x1
            yield frame_bgr[y0:y1, sx0:sx1]

    def _recognize_slot(self, slot_crop: np.ndarray) -> tuple[Optional[Piece], SlotDebug]:
        dbg = SlotDebug()
        if slot_crop.size == 0:
            return None, dbg

        mask = get_binary_mask(slot_crop)
        fill = int(np.count_nonzero(mask))
        dbg.fill_pixels = fill

        slot_area = slot_crop.shape[0] * slot_crop.shape[1]
        if fill < MIN_FILL_PIXELS or fill < MIN_FILL_FRACTION * slot_area:
            return None, dbg
        dbg.has_piece = True

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None, dbg
        r0 = int(rows.argmax())
        r1 = int(len(rows) - rows[::-1].argmax())
        c0 = int(cols.argmax())
        c1 = int(len(cols) - cols[::-1].argmax())
        dbg.bbox = (c0, r0, c1, r1)

        bbox_mask = mask[r0:r1, c0:c1]
        h, w = bbox_mask.shape
        if h < MIN_BBOX_DIM or w < MIN_BBOX_DIM:
            return None, dbg

        # Derive grid dimensions from projection peaks first; only brute-force
        # if no good candidate emerges.
        n_rows_proj = count_clusters(bbox_mask.sum(axis=1))
        n_cols_proj = count_clusters(bbox_mask.sum(axis=0))

        candidates: list[tuple[float, Piece, int, int, float, float, np.ndarray]] = []

        if 1 <= n_rows_proj <= 5 and 1 <= n_cols_proj <= 5:
            self._collect_candidates(
                bbox_mask, n_rows_proj, n_cols_proj, candidates, weight=1.0
            )

        if not candidates:
            for n_r in range(1, 6):
                for n_c in range(1, 6):
                    self._collect_candidates(
                        bbox_mask, n_r, n_c, candidates, weight=0.95
                    )

        if not candidates:
            return None, dbg
        candidates.sort(key=lambda t: t[0], reverse=True)
        score, piece, nr, nc, ch, cw, pattern = candidates[0]

        dbg.best_n_rows = nr
        dbg.best_n_cols = nc
        dbg.best_cell_h = ch
        dbg.best_cell_w = cw
        dbg.best_pattern = pattern
        dbg.best_piece_name = piece.name
        dbg.best_score = score
        dbg.candidates = [
            (s, p.name, r, c) for (s, p, r, c, _, _, _) in candidates[:5]
        ]

        if score < MIN_MATCH_SCORE:
            return None, dbg
        return piece, dbg

    def _collect_candidates(
        self,
        bbox_mask: np.ndarray,
        n_rows: int,
        n_cols: int,
        out: list,
        weight: float,
    ) -> None:
        """Sample bbox at (n_rows, n_cols) and append all template matches."""
        h, w = bbox_mask.shape
        cell_h = h / n_rows
        cell_w = w / n_cols
        ratio = max(cell_h, cell_w) / max(min(cell_h, cell_w), 1e-6)
        if ratio > MAX_CELL_RATIO:
            return
        squareness = min(cell_h, cell_w) / max(cell_h, cell_w)

        pattern = sample_pattern(bbox_mask, n_rows, n_cols, cell_h, cell_w)
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
            score = match_ratio * squareness * weight
            out.append((score, piece, n_rows, n_cols, cell_h, cell_w, pattern))
