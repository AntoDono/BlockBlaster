"""Closed-loop visual servo for auto-play.

Drags one tray piece onto the advisor's target cells with scrcpy
(DOWN → MOVE… → UP) under continuous visual feedback. A PD controller closes
the loop on a 5-point error so the piece lands on target regardless of grab
offset, render lift, or device scaling.

Tuning details live in ``docs/visual-servo.md``.
"""

from blockblaster.control.servo.place import place
from blockblaster.control.servo.types import ServoDebug

__all__ = ["place", "ServoDebug"]
