"""Touch motion and PD control helpers."""

from __future__ import annotations

import time
from typing import Optional

from blockblaster.control.device import Device
from blockblaster.control.servo.config import (
    DERIV_GAIN,
    FINE_STEP_PX,
    GAIN,
    MAX_STEP_PX,
    MOVE_SUBSTEP_MS,
    MOVE_SUBSTEPS,
)


def move_smooth(
    session, to_dev, start_xy, end_xy, *,
    steps: Optional[int] = None,
    substep_ms: Optional[int] = None,
) -> None:
    n = max(1, steps if steps is not None else MOVE_SUBSTEPS)
    pause_ms = substep_ms if substep_ms is not None else MOVE_SUBSTEP_MS
    for i in range(1, n + 1):
        t = i / n
        x = int(round(start_xy[0] + (end_xy[0] - start_xy[0]) * t))
        y = int(round(start_xy[1] + (end_xy[1] - start_xy[1]) * t))
        session.move(*to_dev((x, y)))
        if i < n and pause_ms > 0:
            time.sleep(pause_ms / 1000)


def wait_frame(device: Device, last_fid: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, fid = device.get_latest_with_id()
        if frame is not None and fid != last_fid:
            return frame, fid
        time.sleep(0.005)
    return None, last_fid


def pd_step(
    err_x: int, err_y: int,
    prev_err_x: Optional[int], prev_err_y: Optional[int],
    near_target: bool,
) -> tuple[int, int]:
    derr_x = 0 if prev_err_x is None else err_x - prev_err_x
    derr_y = 0 if prev_err_y is None else err_y - prev_err_y
    ctrl_x = (err_x + DERIV_GAIN * derr_x) / GAIN
    ctrl_y = (err_y + DERIV_GAIN * derr_y) / GAIN
    cap = FINE_STEP_PX if near_target else MAX_STEP_PX
    return (
        max(-cap, min(cap, int(ctrl_x))),
        max(-cap, min(cap, int(ctrl_y))),
    )


def clamp_step(dx: int, dy: int, max_step: int) -> tuple[int, int]:
    """Cap each axis independently (hold-still / micro-nudge mode)."""
    cap = max(0, max_step)
    return (
        max(-cap, min(cap, dx)),
        max(-cap, min(cap, dy)),
    )
