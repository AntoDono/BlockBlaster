"""Mutable state container for the assist pygame app.

Bundling the previously-loose local variables of ``run()`` into a single
dataclass lets us split event handling, auto-play, and the main loop into
separate modules without passing a wall of arguments around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.calibration import CalibrationConfig
from blockblaster.game.board import Board

MODE_GRID   = "grid"
MODE_PIECES = "pieces"


@dataclass
class AppState:
    """All mutable state the event/auto-play layers need to read or write."""

    cfg: CalibrationConfig
    platform: Optional[str] = None
    board: Board = field(default_factory=Board)

    # Calibration UI ---------------------------------------------------------
    calib_mode: str = MODE_GRID
    drag_start: Optional[tuple[int, int]] = None
    drag_cur:   Optional[tuple[int, int]] = None

    # Last frame->screen transform (used to map mouse drags into frame pixels)
    scale:   float = 1.0
    blit_x:  int = 0
    blit_y:  int = 0
    frame_w: int = 1
    frame_h: int = 1

    # Auto-play --------------------------------------------------------------
    auto_enabled: bool = False
    auto_last_executed_frame: int = -1
    auto_busy_until: float = 0.0
    # Sticky gate-reason so app_autoplay can log "why did the loop stop
    # firing" exactly once per consecutive run of the same failure,
    # rather than every analyzer tick.
    auto_last_gate_reason: Optional[str] = None

    # Visual-servo freeze ----------------------------------------------------
    # While a servo placement is in flight we pause the queue CNN + advisor
    # and pin the suggestion / queue / confidences to whatever was on screen
    # at dispatch time.  Board scans keep flowing so the recon panel can
    # show the held piece's solid render drifting into place.
    servo_active: bool = False
    frozen_suggestion: Optional[Suggestion] = None
    frozen_queue: list = field(default_factory=list)
    frozen_confidences: list = field(default_factory=list)

    # Latest held-piece detection from the visual servo, for the recon
    # panel overlay.  Tuple is (top_left_col, top_left_row, piece_rows,
    # piece_cols, score) in board-cell coordinates.  None when no servo
    # is running or the latest frame failed to detect.
    servo_detection: Optional[tuple[int, int, int, int, float]] = None

    # Latest motion mask the matcher saw, cropped to the calibrated
    # board area.  np.uint8 array of shape (fh, fw) with values in
    # {0, 255}.  Published by the servo every iteration; used by the
    # recon panel's debug view (toggled with the "servo dbg" chip / V)
    # so the user can eyeball what the matcher sees in real time.
    servo_debug_mask: Optional[np.ndarray] = None

    # Rolling (frame-to-frame) motion mask, same shape as
    # ``servo_debug_mask``.  Bright only where pixels *changed since
    # last frame* — silent on steady-state row/clear-glow that pollutes
    # the baseline mask.  Used by the servo as a translation gate, and
    # overlaid in magenta on top of the baseline cyan in the debug
    # view so the user can see which part of the mask is actually
    # moving vs which is stale glow.  ``None`` on the very first iter
    # (no prev frame yet) and after the servo finishes.
    servo_debug_mask_rolling: Optional[np.ndarray] = None

    # When True, the recon panel replaces the reconstructed board with
    # the live motion mask scaled to fit.  Detection overlay still
    # renders on top.  Also enables the phone-panel overlay that shows
    # current-piece and target-piece dots + an error vector between them.
    servo_debug_view: bool = False

    # Phone-frame pixel coordinates published by the servo each iter, used
    # by the phone-panel debug overlay.  None means "not currently
    # available".
    #   servo_target_px   — where we're trying to put the piece (bbox centre).
    #   servo_measured_px — where the matcher says the piece currently is.
    servo_target_px:   Optional[tuple[int, int]] = None
    servo_measured_px: Optional[tuple[int, int]] = None

    # Per-cell positions in full-frame pixels.  Same ordering when both
    # lists are the same length (matches ``suggestion.piece.cells``);
    # ``servo_measured_cells`` may be shorter when some cells are
    # occluded.  Used by the debug overlay to draw one dot per cell.
    servo_target_cells:   list[tuple[int, int]] = field(default_factory=list)
    servo_measured_cells: list[tuple[int, int]] = field(default_factory=list)

    # Clickable chip rects returned by draw_controls_panel (keyed by action name)
    control_rects: dict = field(default_factory=dict)

    # Most recent swipe issued by auto-play, stored as frame pixels so the
    # renderer can convert to screen pixels via (scale, blit_x, blit_y).
    # ``last_swipe`` = (src_xy, dst_xy, t_started_ms, duration_ms)
    last_swipe: Optional[tuple[tuple[int, int], tuple[int, int], int, int]] = None
