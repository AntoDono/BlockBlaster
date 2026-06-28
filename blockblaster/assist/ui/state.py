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
