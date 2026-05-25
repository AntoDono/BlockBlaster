"""Pure-function pieces of the PD control loop.

Extracted so the :func:`.place.place` orchestration doesn't drown
them.  Everything here is side-effect-free and trivially unit-testable.
"""

from __future__ import annotations

from typing import Optional

from blockblaster.config.params import DERIV_GAIN, GAIN


def pd_control(
    err: int, prev_err: Optional[int],
) -> tuple[float, int]:
    """One-axis PD step.

    Returns ``(ctrl, derr)`` in finger pixels (uncapped):

    - ``ctrl = p + d`` where ``p = err / GAIN`` and
      ``d = DERIV_GAIN * derr / GAIN``.
    - When ``derr`` flips the sign of ``ctrl`` vs. ``p`` (the D term
      overpowered the P term and pushed us *past* zero), null
      ``ctrl`` out — let the piece coast for one frame.  Prevents the
      spiral overshoot we hit when D was too aggressive near zero.

    ``prev_err is None`` (first iter, or just-reset after a recovery
    override) → ``derr = 0`` so the first step is pure P.
    """
    derr = 0 if prev_err is None else err - prev_err
    p = err / GAIN
    d = DERIV_GAIN * derr / GAIN
    ctrl = p + d
    if (p >= 0) != (ctrl >= 0):
        ctrl = 0.0
    return ctrl, derr


def step_cap(
    err_mag: int,
    near_err_px: int,
    far_err_px: int,
    near_step_px: int,
    far_step_px: int,
) -> int:
    """Distance-adaptive per-iter step ceiling.

    ``|err| ≥ far_err_px`` → ``far_step_px`` (cover ground fast).
    ``|err| ≤ near_err_px`` → ``near_step_px`` (fine alignment).
    In between → linearly interpolated.

    Per-axis, so a piece aligned on x but far on y still gets a fast
    y step without throwing x off.
    """
    if err_mag >= far_err_px:
        return far_step_px
    if err_mag <= near_err_px:
        return near_step_px
    span = max(1, far_err_px - near_err_px)
    t = (err_mag - near_err_px) / span
    return int(round(
        near_step_px + t * (far_step_px - near_step_px),
    ))


# NOTE: an earlier ``recovery_direction(missing_corners, step_px)`` lived
# here that tried to infer the off-board direction from which corner
# anchors went missing.  In practice the inference had the inversion
# backwards: ``matchTemplate`` *saturates* by pushing the reported tl
# toward the *centre* of the board when the piece drifts off-board, so
# the corners that go empty are the ones on the side *opposite* the
# off-board edge.  After getting it wrong in a live run, we replaced
# the directional inference with a simple "push toward the board
# centre" in :func:`.place.place` — robust to whichever side drifted
# off, no inversion to debug.
