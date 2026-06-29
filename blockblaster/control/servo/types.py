"""Shared types for the visual servo package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels


@dataclass
class ServoDebug:
    """Live snapshot of the servo state for the GUI overlay.

    All bboxes are ``(x, y, w, h)`` in frame pixels.
    """
    target_bbox: Optional[Bbox] = None
    measured_bbox: Optional[Bbox] = None
    target_pts: list[tuple[int, int]] = field(default_factory=list)
    measured_pts: list[tuple[int, int]] = field(default_factory=list)
    observe_bbox: Optional[Bbox] = None
    board_aware: bool = False
    unobserved_cells: list[Bbox] = field(default_factory=list)
    finger_px: Optional[tuple[int, int]] = None
    err_px: tuple[int, int] = (0, 0)
    step_px: tuple[int, int] = (0, 0)
    score: float = 0.0
    initial_area_px: int = 0
    current_area_px: int = 0
    locked: bool = False
    status: str = ""


DebugSink = Callable[[Optional[ServoDebug]], None]
LogSink = Callable[[str], None]
