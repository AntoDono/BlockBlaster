"""Bounding-box calibration for the Block Blast grid and piece queue.

All coordinates are stored in **original frame pixels** so they remain valid
regardless of how the frame is scaled/letterboxed inside the pygame window.

JSON format (assist_config.json):
    {
      "grid":  {"fx": 50, "fy": 90, "fw": 280, "fh": 280},
      "queue": {"fx": 40, "fy": 390, "fw": 300, "fh": 80}
    }

Note: the flat single-box format from earlier versions is no longer supported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pygame

CONFIG_PATH = Path("assist_config.json")          # default / backward-compat

_PLATFORM_PATHS: dict[str, Path] = {
    "ios":     Path("assist_config_ios.json"),
    "android": Path("assist_config_android.json"),
}


@dataclass
class CalibrationBox:
    """Bounding box in original frame pixel coordinates."""

    fx: int   # left edge in frame pixels
    fy: int   # top edge in frame pixels
    fw: int   # width in frame pixels
    fh: int   # height in frame pixels

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def to_screen_rect(self, scale: float, blit_x: int, blit_y: int) -> pygame.Rect:
        """Convert to a pygame.Rect in screen coordinates."""
        sx = int(self.fx * scale) + blit_x
        sy = int(self.fy * scale) + blit_y
        sw = int(self.fw * scale)
        sh = int(self.fh * scale)
        return pygame.Rect(sx, sy, sw, sh)

    @staticmethod
    def from_screen(
        px1: int,
        py1: int,
        px2: int,
        py2: int,
        scale: float,
        blit_x: int,
        blit_y: int,
        frame_w: int,
        frame_h: int,
    ) -> "CalibrationBox":
        """Build a CalibrationBox from two screen-space corner points.

        Points are converted to frame coordinates and clamped to the frame bounds.
        """
        sx1, sx2 = sorted([px1, px2])
        sy1, sy2 = sorted([py1, py2])

        fx1 = int((sx1 - blit_x) / scale)
        fy1 = int((sy1 - blit_y) / scale)
        fx2 = int((sx2 - blit_x) / scale)
        fy2 = int((sy2 - blit_y) / scale)

        fx1 = max(0, min(fx1, frame_w))
        fy1 = max(0, min(fy1, frame_h))
        fx2 = max(0, min(fx2, frame_w))
        fy2 = max(0, min(fy2, frame_h))

        return CalibrationBox(fx=fx1, fy=fy1, fw=fx2 - fx1, fh=fy2 - fy1)

    def is_valid(self) -> bool:
        """Return True if the box has non-trivial area."""
        return self.fw > 10 and self.fh > 10


@dataclass
class CalibrationConfig:
    """Holds all calibration boxes for one session.

    Both slots are optional — the user may calibrate one or both independently.
    """

    grid:  Optional[CalibrationBox] = None
    queue: Optional[CalibrationBox] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Optional[Path] = None,
        platform: Optional[str] = None,
    ) -> None:
        """Persist to JSON.

        Priority: explicit *path* > *platform* lookup > default :data:`CONFIG_PATH`.
        """
        if path is None:
            path = _PLATFORM_PATHS.get(platform or "", CONFIG_PATH)
        data: dict = {}
        if self.grid is not None:
            data["grid"] = asdict(self.grid)
        if self.queue is not None:
            data["queue"] = asdict(self.queue)
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(
        path: Optional[Path] = None,
        platform: Optional[str] = None,
    ) -> "CalibrationConfig":
        """Load from JSON.  Returns an empty config on any error.

        Priority: explicit *path* > *platform* lookup > default :data:`CONFIG_PATH`.
        Falls back to the legacy default path so existing configs are not lost.
        """
        if path is None:
            path = _PLATFORM_PATHS.get(platform or "", CONFIG_PATH)
        cfg = CalibrationConfig()
        try:
            data = json.loads(path.read_text())
            if "grid" in data:
                cfg.grid = CalibrationBox(**data["grid"])
            if "queue" in data:
                cfg.queue = CalibrationBox(**data["queue"])
        except Exception:
            pass
        return cfg
