"""Debug diagnostics for the piece recognizer (overlays, summary reports).

Kept separate so :mod:`piece_recognizer` can stay focused on the recognition
pipeline.  All rendering helpers here are pure functions of ``(crop, dbg)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from blockblaster.game.pieces import Piece


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


def render_overlay(slot_crop: np.ndarray, dbg: SlotDebug) -> np.ndarray:
    """Draw bbox + best (n_rows × n_cols) sub-grid on top of the slot."""
    out = slot_crop.copy()
    if not dbg.has_piece:
        cv2.putText(out, "EMPTY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    x0, y0, x1, y1 = dbg.bbox
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)

    if dbg.best_n_rows > 0 and dbg.best_n_cols > 0:
        for r in range(1, dbg.best_n_rows):
            yy = int(y0 + r * dbg.best_cell_h)
            cv2.line(out, (x0, yy), (x1 - 1, yy), (255, 255, 0), 1)
        for c in range(1, dbg.best_n_cols):
            xx = int(x0 + c * dbg.best_cell_w)
            cv2.line(out, (xx, y0), (xx, y1 - 1), (255, 255, 0), 1)

        if dbg.best_pattern is not None:
            for r in range(dbg.best_n_rows):
                for c in range(dbg.best_n_cols):
                    if dbg.best_pattern[r, c]:
                        cy = int(y0 + (r + 0.5) * dbg.best_cell_h)
                        cx = int(x0 + (c + 0.5) * dbg.best_cell_w)
                        cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)

    label = (
        f"{dbg.best_piece_name or '?'} "
        f"({dbg.best_n_rows}x{dbg.best_n_cols}) score={dbg.best_score:.2f}"
    )
    cv2.putText(out, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def format_slot_report(idx: int, piece: Optional[Piece], dbg: SlotDebug) -> str:
    """Format one slot's diagnostics as a human-readable text block."""
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
