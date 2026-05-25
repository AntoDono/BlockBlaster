"""Closed-loop visual-servo placement.

Detection
=========
Colour- and palette-invariant.  Cache the grayscale board crop just
before DOWN as a baseline, then per frame compute
``cv2.absdiff(current_gray, baseline_gray)``, threshold to a binary
"this pixel moved" mask (morph-closed), and run
``cv2.matchTemplate`` of the known piece silhouette against that
motion mask to get a rigid initial pose.  The piece is the only thing
moving on the board, so the diff lights up exactly its rendered
footprint regardless of colour, translucency, or ghost-preview noise
from the game itself.  No tracker state between frames, no plant-gain
learning, no coarse open-loop jump, no fallbacks.

Per-frame measurement: 5 anchors
================================
Instead of one centroid we sample 5 anchors per frame and pair them
positionally with their target positions:

  - TL / TR / BL / BR: the extreme moving pixel in each corner
    direction of the corresponding extreme cell of the piece
    silhouette (e.g. for TL, the topmost-leftmost moving pixel in the
    top-row leftmost cell).  Anchored to crisp silhouette edges →
    robust to interior mass distribution.
  - C: centre-of-mass of motion in the most-central cell of the
    silhouette (elbow of an L, middle cell of a line, etc.).  Robust
    to corner occlusions which would bias corner anchors.

Anchors whose cell coverage falls below ``CELL_MIN_COVERAGE`` are
dropped from the per-frame error.  The controller's error is the mean
of ``(target_i − measured_i)`` over the visible anchors, so partial
occlusions (board edges, score popups, animation flashes) don't bias
the average.

Gesture
=======
1. DOWN on the queue slot (centre nudged up by ``GRAB_Y_NUDGE_PX``),
   hold ``HOLD_MS`` so Block Blast registers the long-press grab.
2. Pre-lift diagonally to ``(board_centre_x, slot_y − INITIAL_LIFT_PX)``
   so wide pieces grabbed from edge slots don't hang off the board
   and lose detection.
3. Wait up to ``PRELIFT_CONFIRM_S`` for the matcher to confirm the
   piece is rendered (≥1 anchor + score ≥ ``LOCK_SCORE_MIN``).  Abort
   if it never confirms — better than blindly servoing a piece we
   can't see.
4. PD loop (below) until lock or budget.
5. ``PRE_LIFT_MS`` settle, then UP.

PD loop
=======
P scales with error (``err / GAIN``); D dampens by anticipating the
piece's motion (``DERIV_GAIN * derr / GAIN``).  If D flips the sign
of the P term it's nulled (let the piece coast one frame) — this
prevents the spiral overshoot we hit when D was too aggressive near
zero error.

The per-iter step ceiling is **distance-adaptive**:

  |err| ≥ far_err_px   → cap = max_step_far_px   (cover ground)
  |err| ≤ near_err_px  → cap = max_step_near_px  (fine alignment)
  in between           → linearly interpolated

Per-axis, so a piece aligned on x but far on y still gets a fast y
step without throwing x off.  ``max_step_far_px`` is bounded by Block
Blast's drag follower lag — go too fast and the next-frame match
reads a stale position and the PD overshoots.

Release
=======
The UP commits the placement, so the gate is conservative:

  (tight_lock OR transit_lock) AND score ≥ LOCK_SCORE_MIN
                                AND paired ≥ LOCK_MIN_ANCHORS

  tight_lock:   |err_x| ≤ LOCK_TOL_PX AND |err_y| ≤ LOCK_TOL_PX
  transit_lock: each axis was inside LOCK_TOL_PX at some point this
                run AND is still within 2× tol now (catches a
                diagonal pass-through where the axes peak on
                different frames).

If the matcher loses the piece for ``MAX_NO_PIECE_FRAMES``
consecutive iters, abort.  If the ``MAX_LOOP_S`` budget elapses
without lock, lift in place rather than dragging back to the queue.

Pre-clear glow early release
============================
When the held piece is over a placement that would clear a row or
column, Block Blast pre-renders a glow over the cells that would
clear.  That glow paints the motion-diff mask far beyond the piece's
own footprint and can fool the template matcher into reporting a
stale position.  The piece is, by definition, on an optimal placement
when the glow appears, so we just release.  The check sits *before*
the matcher-driven PD logic each iter, so it fires even when the
glow has already confused the matcher; persistence (``GLOW_HOLD_S``)
prevents transient flashes from accidentally committing.

Off-board corner recovery
=========================
When the corner anchors that *should* be visible at the target stop
reporting (their cell windows have no motion to detect because the
piece is hanging off-board on that side), we override the PD step
with a fixed inward nudge until the missing corners reappear.
Direction is inferred from which side's corners are AWOL.  Hold gate
(``OFF_BOARD_HOLD_S``) filters transient occlusions.

All tunables live in :mod:`blockblaster.config.params`.

Module layout
=============
- :mod:`.detection` — motion mask, template, piece anchors, locate.
- :mod:`.control`   — pure PD step, step cap, recovery direction.
- :mod:`.io`        — touch-driver and frame-acquisition helpers.
- :mod:`.place`     — top-level ``place()`` orchestration.
"""

from __future__ import annotations

from blockblaster.config.params import GRAB_Y_NUDGE_PX

from .place import place

__all__ = ["place", "GRAB_Y_NUDGE_PX"]
