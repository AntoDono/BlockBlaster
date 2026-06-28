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
