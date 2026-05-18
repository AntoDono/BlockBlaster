"""Two-mode P controller + Y-axis anti-windup.

The held piece's plant gain is >1 and grows with drag distance, so a
single fixed gain either oscillates (gain too high near the target) or
stalls (gain too low for far errors).  We split correction sizing into
two regimes:

* **Coarse (|raw_err| > FINE_THRESHOLD_PX)** — take a fixed
  ``STEP_CLAMP_PX`` stride toward the target every iteration.  The
  closing speed is ``plant_g × STEP_CLAMP_PX`` px / iter, which is
  fast enough to traverse the whole board within the loop budget.
* **Fine (|raw_err| ≤ FINE_THRESHOLD_PX)** — *invert the plant*: to
  move the piece ``raw`` px, command the finger ``raw / plant_g`` px
  (then de-rate by ``FINE_SAFETY_FACTOR``).  Naturally shrinks the
  step when the plant gain has grown — exactly the high-offset
  overshoot case the original fixed P controller couldn't handle.
"""

from __future__ import annotations

from blockblaster.control.visual_servo.tunables import (
    FINE_SAFETY_FACTOR,
    FINE_THRESHOLD_PX,
    FINGER_OVERTRAVEL_Y,
    PLANT_GAIN_MIN,
    STEP_CLAMP_PX,
)


def axis_step(raw: int, plant_g: float) -> int:
    """Compute the per-axis finger displacement for one servo iteration.

    Parameters
    ----------
    raw:
        Signed pixel error (target − current piece anchor) on one axis.
    plant_g:
        Current online estimate of (piece px / finger px) on that axis.
        Used only in the fine region; the coarse region is plant-blind by
        design so a bad early estimate can't stall progress on big errors.
    """
    if abs(raw) > FINE_THRESHOLD_PX:
        return STEP_CLAMP_PX if raw > 0 else -STEP_CLAMP_PX
    step = int(round(FINE_SAFETY_FACTOR * raw / max(plant_g, PLANT_GAIN_MIN)))
    return max(-STEP_CLAMP_PX, min(STEP_CLAMP_PX, step))


def clamp_finger_y(new_y: int, target_y: int) -> int:
    """Y-axis anti-windup: forbid the finger from drifting too far below
    the target row.

    Block Blast renders the held piece *above* the finger with a growing
    offset, but only up to a point — once the finger has slid past the
    bottom of the board, additional downward commands stop translating
    into piece motion and just risk losing the grab.  We cap the finger
    Y at ``target_y + FINGER_OVERTRAVEL_Y``.  See the constant comment
    for the failure mode this prevents.
    """
    max_y = target_y + FINGER_OVERTRAVEL_Y
    return min(new_y, max_y)
