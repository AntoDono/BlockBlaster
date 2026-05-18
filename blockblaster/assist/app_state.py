"""Mutable state container for the assist pygame app.

Bundling the previously-loose local variables of ``run()`` into a single
dataclass lets us split event handling, auto-play, and the main loop into
separate modules without passing a wall of arguments around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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

    # Visual-servo freeze ----------------------------------------------------
    # While a servo placement is in flight we pause the queue CNN + advisor
    # and pin the suggestion / queue / confidences to whatever was on screen
    # at dispatch time.  Board scans keep flowing so the recon panel can
    # show the ghost piece drifting into place.
    servo_active: bool = False
    frozen_suggestion: Optional[Suggestion] = None
    frozen_queue: list = field(default_factory=list)
    frozen_confidences: list = field(default_factory=list)

    # Clickable chip rects returned by draw_controls_panel (keyed by action name)
    control_rects: dict = field(default_factory=dict)

    # Most recent swipe issued by auto-play, stored as frame pixels so the
    # renderer can convert to screen pixels via (scale, blit_x, blit_y).
    # ``last_swipe`` = (src_xy, dst_xy, t_started_ms, duration_ms)
    last_swipe: Optional[tuple[tuple[int, int], tuple[int, int], int, int]] = None
