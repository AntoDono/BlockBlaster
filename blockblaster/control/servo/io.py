"""Touch-driver and frame-acquisition helpers for the visual servo."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from blockblaster.config.params import MOVE_SUBSTEP_MS, MOVE_SUBSTEPS
from blockblaster.control.device import Device


def move_smooth(
    session,
    to_dev: Callable[[tuple[int, int]], tuple[int, int]],
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
) -> None:
    """Interpolate a single move into ``MOVE_SUBSTEPS`` touch events.

    ``start_xy`` and ``end_xy`` are in frame pixels.  Spacing the
    intermediate moves by ``MOVE_SUBSTEP_MS`` lets Block Blast's drag
    follower render the piece continuously instead of jumping, which
    keeps the next-frame matcher reading accurate.
    """
    steps = max(1, MOVE_SUBSTEPS)
    for i in range(1, steps + 1):
        t = i / steps
        x = int(round(start_xy[0] + (end_xy[0] - start_xy[0]) * t))
        y = int(round(start_xy[1] + (end_xy[1] - start_xy[1]) * t))
        session.move(*to_dev((x, y)))
        if i < steps and MOVE_SUBSTEP_MS > 0:
            time.sleep(MOVE_SUBSTEP_MS / 1000)


def wait_frame(
    device: Device, last_fid: int, timeout_s: float,
) -> tuple[Optional[np.ndarray], int]:
    """Block until a fresh frame (``fid != last_fid``) shows up, or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_fid:
            return frame, fid
        time.sleep(0.005)
    return None, last_fid
