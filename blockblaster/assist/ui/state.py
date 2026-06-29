"""Mutable state container for the assist pygame app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    platform: Optional[str] = None
    frame_w: int = 1
    frame_h: int = 1
    control_rects: dict = field(default_factory=dict)
    autoplay_on: bool = False
    servo_busy: bool = False
    auto_next_after: float = 0.0
    show_debug: bool = False
    servo_debug: Optional[object] = None  # control.servo.ServoDebug while active
    recalibrate_request: bool = False

    # Manual board editing: drag a box on the phone panel to override the
    # auto-detected board region. ``phone_map`` is the live (scale, bx, by) that
    # maps frame px → phone-panel screen px (published each draw).
    edit_board: bool = False
    board_override: Optional[tuple[int, int, int, int]] = None  # (x,y,w,h) frame px
    phone_map: tuple[float, int, int] = (1.0, 0, 0)
    drag_start_frame: Optional[tuple[int, int]] = None  # in-progress drag anchor
    drag_cur_frame: Optional[tuple[int, int]] = None
