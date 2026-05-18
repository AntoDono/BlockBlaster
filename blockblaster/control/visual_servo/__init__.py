"""Closed-loop visual-servo placer for Block Blast.

Press the finger on a queue slot, drag toward the planned cells while
watching the held piece's on-board render, lift when it matches.  See
:doc:`docs/visual-servo.md` for the algorithm walk-through.

Module layout
-------------
* :mod:`.tunables`   — every magic constant, each with a comment.
* :mod:`.plant_gain` — online estimator + persistent learned cache.
* :mod:`.controller` — two-mode P controller + Y anti-windup.
* :mod:`.detection`  — held-piece detection + lock check.
* :mod:`.placer`     — orchestration / public entrypoint.

The package re-exports the public surface so callers don't need to know
the internal layout.
"""

from blockblaster.control.visual_servo.placer import place_with_servo
from blockblaster.control.visual_servo.tunables import (
    GRAB_Y_NUDGE_PX as _GRAB_Y_NUDGE_PX,
    ServoResult,
)

__all__ = ["place_with_servo", "ServoResult", "_GRAB_Y_NUDGE_PX"]
