"""Mutable state container for the assist pygame app."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import pygame

from blockblaster.assist.ui.log_buffer import LOG_MAX_LINES

if TYPE_CHECKING:
    from blockblaster.control.servo import ServoDebug


@dataclass
class AppState:
    platform: Optional[str] = None
    frame_w: int = 1
    frame_h: int = 1
    control_rects: dict[str, pygame.Rect] = field(default_factory=dict)
    autoplay_on: bool = False
    servo_busy: bool = False
    auto_next_after: float = 0.0
    await_fresh_suggestion: bool = False
    placed_suggestion_key: Optional[tuple[int, int, int, int]] = None
    # Last suggestion we already wrote a [plan] line for; reset to None on
    # recalibrate so the next stable plan is re-logged.
    logged_plan_key: Optional[tuple[int, int, int, int]] = None
    show_debug: bool = False
    servo_debug: Optional[ServoDebug] = None
    reset_analysis_request: bool = False
    await_pre_move_scan: bool = False
    pre_move_scan_after: float = 0.0
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_MAX_LINES))
    log_scroll: int = 0  # wrapped lines scrolled up from the bottom (0 = follow tail)
    log_rect: pygame.Rect = field(default_factory=pygame.Rect)

    # Manual board editing: drag a box on the phone panel to override the
    # auto-detected board region. ``phone_map`` is the live (scale, bx, by)
    # that maps frame px → phone-panel screen px (published each draw).
    edit_board: bool = False
    board_override: Optional[tuple[int, int, int, int]] = None  # (x,y,w,h) frame px
    phone_map: tuple[float, int, int] = (1.0, 0, 0)
    drag_start_frame: Optional[tuple[int, int]] = None
    drag_cur_frame: Optional[tuple[int, int]] = None
