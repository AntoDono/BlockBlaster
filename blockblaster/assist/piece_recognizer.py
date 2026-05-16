"""Piece recognition for the Block Blast queue via sub-grid sampling.

Pipeline per slot:
  1. Crop one of the 3 equal horizontal thirds from the queue bounding box.
  2. HSV-threshold the crop → binary mask.
  3. If the mask has too few filled pixels → empty slot → return None.
  4. Find the tight bounding box of the mask (this trims out the empty padding
     around a centred piece, regardless of where the piece sits inside the slot).
  5. Try every plausible (n_rows, n_cols) ∈ {1..5} × {1..5} sub-division of the
     bbox.  Skip sub-divisions whose implied cell aspect is far from square.
  6. For each candidate, sample each sub-cell's mean mask value → boolean
     pattern.  The pattern must be tight (filled cells touch every edge).
  7. For each pattern, compare against all 32 piece templates that share its
     shape; score = (cell-match ratio) × (cell-squareness preference).
  8. Return the highest-scoring piece (or None if no candidate cleared the
     minimum confidence).

Why this beats blob detection: it never tries to separate adjacent cells.
Touching cells, fragmented cells from highlights/shadows, and weird mask
gradients are all OK — every candidate sub-division re-tiles the bbox cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.game.pieces import PIECES, Piece

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


@dataclass
class SlotDebug:
    """Diagnostic info for one slot — populated when debug=True."""

    has_piece: bool = False
    fill_pixels: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # x0, y0, x1, y1 in slot coords
    best_n_rows: int = 0
    best_n_cols: int = 0
    best_cell_h: float = 0.0
    best_cell_w: float = 0.0
    best_pattern: Optional[np.ndarray] = None
    best_piece_name: Optional[str] = None
    best_score: float = 0.0
    candidates: list[tuple[float, str, int, int]] = field(default_factory=list)


class PieceRecognizer:
    """Recognises which of the 32 Block Blast pieces occupy each queue slot.

    By default this uses a CNN classifier trained on synthetic data
    (``piece_cnn.pt``).  If the CNN weights are missing or fail to load it
    transparently falls back to the legacy projection / template-matching
    heuristic so the assist GUI still works.
    """

    def __init__(self) -> None:
        # Pre-compute tight boolean grid for every piece (template lookup table)
        self._templates: list[tuple[Piece, np.ndarray]] = [
            (p, self._tight(p.to_grid())) for p in PIECES
        ]
        # Try to load the trained CNN; if missing we fall back to heuristics.
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
        slot_crops = list(self._iter_slot_crops(frame_bgr, queue_box))
        if self._cnn is not None and self._cnn.is_ready:
            return [p for p, _ in self._cnn.classify_slots(slot_crops)]
        # Heuristic fallback
        return [p for p, _ in self._recognize_queue_with_debug(frame_bgr, queue_box)]

    def save_debug(
        self,
        frame_bgr: np.ndarray,
        queue_box: CalibrationBox,
        out_dir: Path | str = "assist_debug",
    ) -> Path:
        """Run recognition and dump per-slot diagnostic images + summary.

        Returns the output directory path.  Files written per slot:
            slot_{i}_crop.png        — raw BGR slot crop
            slot_{i}_mask.png        — binary HSV mask
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

            mask = self._get_binary_mask(slot_crop)
            cv2.imwrite(str(out_path / f"{slot_name}_mask.png"), mask)

            overlay = self._render_overlay(slot_crop, dbg)
            cv2.imwrite(str(out_path / f"{slot_name}_overlay.png"), overlay)

            report.append(self._format_slot_report(i, piece, dbg))

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
        return [self._recognize_slot(crop) for crop in self._iter_slot_crops(frame_bgr, queue_box)]

    def _iter_slot_crops(
        self,
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

        mask = self._get_binary_mask(slot_crop)
        fill = int(np.count_nonzero(mask))
        dbg.fill_pixels = fill

        slot_area = slot_crop.shape[0] * slot_crop.shape[1]
        if fill < MIN_FILL_PIXELS or fill < MIN_FILL_FRACTION * slot_area:
            return None, dbg
        dbg.has_piece = True

        # Tight bbox of mask
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

        # Derive grid dimensions from projection peaks (preferred — the cells
        # of the piece produce distinct plateaus in the row/col mask sums).
        n_rows_proj = self._count_clusters(bbox_mask.sum(axis=1))
        n_cols_proj = self._count_clusters(bbox_mask.sum(axis=0))

        candidates: list[tuple[float, Piece, int, int, float, float, np.ndarray]] = []

        # Preferred path: projection-detected grid dims (each cell of the piece
        # produces a distinct plateau in the row/col mask sums, separated by
        # gaps where the piece has no cell).
        if 1 <= n_rows_proj <= 5 and 1 <= n_cols_proj <= 5:
            self._collect_candidates(
                bbox_mask, n_rows_proj, n_cols_proj, candidates, weight=1.0
            )

        # Fallback: brute-force grid dims, used only when projection produced
        # no valid candidate (e.g. cells touch with no visible separator).
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
        """Sample bbox at (n_rows, n_cols) and append all template matches to `out`."""
        h, w = bbox_mask.shape
        cell_h = h / n_rows
        cell_w = w / n_cols
        ratio = max(cell_h, cell_w) / max(min(cell_h, cell_w), 1e-6)
        if ratio > MAX_CELL_RATIO:
            return
        squareness = min(cell_h, cell_w) / max(cell_h, cell_w)

        pattern = self._sample_pattern(bbox_mask, n_rows, n_cols, cell_h, cell_w)
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

    @staticmethod
    def _count_clusters(profile: np.ndarray, threshold_frac: float = 0.3) -> int:
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

    # ------------------------------------------------------------------
    # Mask + sampling
    # ------------------------------------------------------------------

    def _get_binary_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 binary mask of foreground (piece) pixels.

        Uses background subtraction: samples the slot's border ring (which is
        always background because pieces are centred with padding), then masks
        any pixel that differs sufficiently from that background colour.
        """
        h, w = crop_bgr.shape[:2]
        b = min(BG_BORDER_PX, h // 4, w // 4)
        if b < 1:
            return np.zeros((h, w), dtype=np.uint8)

        # Sample background BGR from a thin ring around the slot edge
        border_pixels = np.concatenate([
            crop_bgr[:b, :].reshape(-1, 3),
            crop_bgr[-b:, :].reshape(-1, 3),
            crop_bgr[b:-b, :b].reshape(-1, 3),
            crop_bgr[b:-b, -b:].reshape(-1, 3),
        ])
        bg_color = np.median(border_pixels, axis=0).astype(np.int16)

        # Sum-of-channel-abs-diff per pixel (range 0..765)
        diff = np.abs(crop_bgr.astype(np.int16) - bg_color).sum(axis=2)
        mask = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255

        # Morphological open to remove salt noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def _sample_pattern(
        bbox_mask: np.ndarray,
        n_rows: int,
        n_cols: int,
        cell_h: float,
        cell_w: float,
    ) -> Optional[np.ndarray]:
        """Slice bbox_mask into n_rows × n_cols cells; return bool pattern."""
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

    # ------------------------------------------------------------------
    # Debug rendering
    # ------------------------------------------------------------------

    def _render_overlay(self, slot_crop: np.ndarray, dbg: SlotDebug) -> np.ndarray:
        """Draw bbox + best (n_rows × n_cols) sub-grid on top of the slot."""
        out = slot_crop.copy()
        if not dbg.has_piece:
            cv2.putText(out, "EMPTY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return out

        x0, y0, x1, y1 = dbg.bbox
        # Yellow bbox
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)

        # Sub-grid lines (cyan)
        if dbg.best_n_rows > 0 and dbg.best_n_cols > 0:
            for r in range(1, dbg.best_n_rows):
                yy = int(y0 + r * dbg.best_cell_h)
                cv2.line(out, (x0, yy), (x1 - 1, yy), (255, 255, 0), 1)
            for c in range(1, dbg.best_n_cols):
                xx = int(x0 + c * dbg.best_cell_w)
                cv2.line(out, (xx, y0), (xx, y1 - 1), (255, 255, 0), 1)

            # Mark filled cells with green dots
            if dbg.best_pattern is not None:
                for r in range(dbg.best_n_rows):
                    for c in range(dbg.best_n_cols):
                        if dbg.best_pattern[r, c]:
                            cy = int(y0 + (r + 0.5) * dbg.best_cell_h)
                            cx = int(x0 + (c + 0.5) * dbg.best_cell_w)
                            cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)

        # Label at top
        label = (
            f"{dbg.best_piece_name or '?'} "
            f"({dbg.best_n_rows}x{dbg.best_n_cols}) score={dbg.best_score:.2f}"
        )
        cv2.putText(out, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _format_slot_report(idx: int, piece: Optional[Piece], dbg: SlotDebug) -> str:
        lines = [f"=== Slot {idx + 1} ==="]
        lines.append(f"  fill_pixels   : {dbg.fill_pixels}")
        lines.append(f"  has_piece     : {dbg.has_piece}")
        if dbg.has_piece:
            lines.append(f"  bbox (x,y→x,y): {dbg.bbox}")
            lines.append(
                f"  best fit      : {dbg.best_n_rows} rows × {dbg.best_n_cols} cols  "
                f"(cell {dbg.best_cell_h:.1f} × {dbg.best_cell_w:.1f}px)"
            )
            if dbg.best_pattern is not None:
                pretty = "\n".join(
                    "                  " + "".join("X" if v else "." for v in row)
                    for row in dbg.best_pattern
                )
                lines.append("  pattern       :\n" + pretty)
            lines.append(f"  best score    : {dbg.best_score:.3f}")
            lines.append("  top candidates:")
            for s, name, r, c in dbg.candidates:
                lines.append(f"    {s:.3f}  {name:<10}  ({r}×{c})")
        lines.append(f"  → result      : {piece.name if piece else 'None'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tight(grid: np.ndarray) -> np.ndarray:
        """Remove all-False border rows and columns from a bool grid."""
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)
        if not rows.any():
            return grid
        r0, r1 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
        c0, c1 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())
        return grid[r0:r1, c0:c1]
